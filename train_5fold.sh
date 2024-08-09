#!/bin/bash

# 指定Python环境
PYTHON_ENV=python3

# 指定参数
DATA_DIR="/mnt/usb5/jijianxin/new_wsi/HMU_ALL_512_PT"
BATCH_SIZE=1
DEPTH=2  # 设置Transformer的深度
NUM_FEATURES=512  # 输入特征的维度
DROP_RATE=0.1  # dropout rate
DROP_PATH_RATE=0.2  # drop path rate
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
EARLY_STOPPING_PATIENCE=20  # early stopping patience
LOG_FILE="training.log"
ACCUMULATION_STEPS=4  # 梯度累积步骤
SAVE_DIR="save_dir"  # 保存结果的目录

# 自动混合精度
USE_AMP="--use_amp"

# 创建保存目录
mkdir -p $SAVE_DIR

# 运行五折交叉验证
for FOLD in {0..4}
do
    echo "Running fold $FOLD..."
    
    TRAIN_DF="splits/filtered/train_set_$FOLD.csv"
    VAL_DF="splits/filtered/val_set_$FOLD.csv"
    
    $PYTHON_ENV main_new.py \
        --df_dir "splits/filtered" \
        --data_dir $DATA_DIR \
        --batch_size $BATCH_SIZE \
        --depth $DEPTH \
        --num_features $NUM_FEATURES \
        --drop_rate $DROP_RATE \
        --drop_path_rate $DROP_PATH_RATE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --log_file "${LOG_FILE}_fold_${FOLD}.log" \
        --train_csv $TRAIN_DF \
        --val_csv $VAL_DF \
        --accumulation_steps $ACCUMULATION_STEPS \
        --save_dir "${SAVE_DIR}/fold_${FOLD}" \
        $USE_AMP
    
    echo "Fold $FOLD completed."
done

echo "All folds completed."
