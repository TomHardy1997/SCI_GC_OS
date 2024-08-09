import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import logging
import argparse
import ast

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SwinPrognosisDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = pd.read_csv(df)
        self.patient = self.df['case_id']
        self.label = self.df['label']
        self.censor = self.df['censor']
        self.time = self.df['survival_months']
        self.wsi = self.df['slide_id'].apply(lambda x: ast.literal_eval(x.strip()))
        self.data_dir = data_dir
        logging.info("SwinPrognosisDataset initialized with {} samples".format(len(self.patient)))
    
    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):
        patient = self.patient[idx]
        label = self.label[idx]
        censor = self.censor[idx]
        sur_time = self.time[idx]
        slide_ids = self.wsi[idx]
        path_features = []
        for slide_id in slide_ids:
            slide_id = slide_id.strip()
            wsi_path = os.path.join(self.data_dir, slide_id)
            try:
                wsi_bag = torch.load(wsi_path, weights_only=True)
                path_features.append(wsi_bag)
            except FileNotFoundError:
                logging.error(f"File not found: {wsi_path}")
                continue
            except RuntimeError as e:
                logging.error(f"Error loading file {wsi_path}: {e}")
                continue
        if path_features:
            path_features = torch.cat(path_features, dim=0)
        else:
            path_features = torch.tensor([])
        return path_features, label, sur_time, censor, patient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tumor and SwinPrognosis Dataset")
    parser.add_argument('--df', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    args = parser.parse_args()

    dataset = SwinPrognosisDataset(df=args.df, data_dir=args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)

    for data in data_loader:
        print(data)
        break
