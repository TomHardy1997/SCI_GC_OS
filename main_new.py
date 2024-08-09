import os
import torch
import pandas as pd
import logging
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from dataset_new import SwinPrognosisDataset
from transformer import Transformer
from loss_func import NLLSurvLoss
import numpy as np
from sksurv.metrics import concordance_index_censored


def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])


class ClassifierLightning(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, weight_decay, save_dir):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_dir = save_dir  # 添加保存目录
        self.val_all_risk_scores = []
        self.val_all_censorships = []
        self.val_all_event_times = []

        self.train_all_risk_scores = []
        self.train_all_censorships = []
        self.train_all_event_times = []

        self.best_val_c_index = 0.0  # 用于存储最佳的 C-Index

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        path_features, label, sur_time, censor, _ = batch
        outputs = self(path_features)
        loss = self.criterion(h=outputs, y=label, t=sur_time, c=censor)
        self.log('train_loss', loss, sync_dist=True)

        # Collecting data for C-Index calculation
        with torch.no_grad():
            train_hazards = torch.sigmoid(outputs)
            train_survival = torch.cumprod(1 - train_hazards, dim=1)
            train_risk = -torch.sum(train_survival, dim=1).detach().cpu().numpy()

        # Save outputs to the instance attributes
        self.train_all_risk_scores.append(train_risk)
        self.train_all_censorships.append(censor.cpu().numpy())
        self.train_all_event_times.append(sur_time.cpu().numpy())

        return loss

    def validation_step(self, batch, batch_idx):
        path_features, label, sur_time, censor, _ = batch
        outputs = self(path_features)
        loss = self.criterion(h=outputs, y=label, t=sur_time, c=censor)
        self.log('val_loss', loss, sync_dist=True)

        # Collecting data for C-Index calculation
        with torch.no_grad():
            test_hazards = torch.sigmoid(outputs)
            test_survival = torch.cumprod(1 - test_hazards, dim=1)
            test_risk = -torch.sum(test_survival, dim=1).detach().cpu().numpy()

        # Save outputs to the instance attributes
        self.val_all_risk_scores.append(test_risk)
        self.val_all_censorships.append(censor.cpu().numpy())
        self.val_all_event_times.append(sur_time.cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        # Flatten lists for training data
        if len(self.train_all_risk_scores) == 0:
            return

        train_all_risk_scores = np.concatenate(self.train_all_risk_scores)
        train_all_censorships = np.concatenate(self.train_all_censorships)
        train_all_event_times = np.concatenate(self.train_all_event_times)

        # Calculate C-Index for training data
        c_index = concordance_index_censored((1 - train_all_censorships).astype(bool), train_all_event_times, train_all_risk_scores)[0]
        self.log('train_c_index', c_index, sync_dist=True)
        logging.info(f'Training C-Index: {c_index}')

        # Clear the lists for the next epoch
        self.train_all_risk_scores = []
        self.train_all_censorships = []
        self.train_all_event_times = []

    def on_validation_epoch_end(self):
        # Flatten lists for validation data
        if len(self.val_all_risk_scores) == 0:
            return  # If there are no samples, return early to avoid errors

        val_all_risk_scores = np.concatenate(self.val_all_risk_scores)
        val_all_censorships = np.concatenate(self.val_all_censorships)
        val_all_event_times = np.concatenate(self.val_all_event_times)

        # Check if all samples are censored
        if np.all(val_all_censorships == 1):
            logging.warning("All samples are censored, C-Index cannot be calculated.")
            self.log('val_c_index', 0.0, sync_dist=True)
        else:
            # Calculate C-Index for validation data
            c_index = concordance_index_censored((1 - val_all_censorships).astype(bool), val_all_event_times, val_all_risk_scores)[0]
            self.log('val_c_index', c_index, sync_dist=True)
            logging.info(f'Validation C-Index: {c_index}')

            # 如果当前C-Index是最好的，保存模型参数和C-Index到CSV
            if c_index > self.best_val_c_index:
                self.best_val_c_index = c_index

                # 确保保存目录存在
                os.makedirs(self.save_dir, exist_ok=True)

                # 保存模型参数
                best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f'Best model saved at: {best_model_path}')

                # 保存C-Index到CSV
                best_c_index_csv_path = os.path.join(self.save_dir, 'best_c_index.csv')
                pd.DataFrame({'best_val_c_index': [c_index]}).to_csv(best_c_index_csv_path, index=False)
                logging.info(f'Best C-Index saved at: {best_c_index_csv_path}')

        # Clear the lists for the next epoch
        self.val_all_risk_scores = []
        self.val_all_censorships = []
        self.val_all_event_times = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def main():
    parser = argparse.ArgumentParser(description="Training Transformer for Survival Analysis")
    parser.add_argument('--df_dir', type=str, default='splits/filtered', help='Directory containing the CSV files for data splits')
    parser.add_argument('--data_dir', type=str, default='/mnt/usb5/jijianxin/', help='Directory containing the data files')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the transformer')  # Reduced depth
    parser.add_argument('--num_features', type=int, default=512, help='Input dimension')  # Reduced input dimension
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='Drop path rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--log_file', type=str, default='training.log', help='Log file path')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')  # Increased accumulation steps
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the best model and C-Index')
    args = parser.parse_args()

    setup_logging(args.log_file)

    model = Transformer(
        num_classes=4,
        input_dim=512,
        dim=256,  # Reduced model dimension
        depth=2,
        heads=8,
        mlp_dim=128,  # Reduced MLP dimension
        pool='cls',
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    )

    criterion = NLLSurvLoss(alpha=0.4)

    train_dataset = SwinPrognosisDataset(args.train_csv, data_dir=args.data_dir)
    val_dataset = SwinPrognosisDataset(args.val_csv, data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    classifier = ClassifierLightning(model, criterion, args.learning_rate, args.weight_decay, args.save_dir)
    wandb_logger = WandbLogger(project='project_name', name=args.log_file, save_dir='wandb_logs', mode='offline')
    csv_logger = CSVLogger(save_dir='csv_logs', name=args.log_file)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.save_dir,  # 保存模型的路径使用 save_dir 参数
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        logger=[wandb_logger, csv_logger],
        precision='16-mixed' if args.use_amp else 32,
        accelerator='auto',
        devices='auto',
        max_epochs=300,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulation_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(classifier, train_loader, val_loader)

if __name__ == "__main__":
    main()