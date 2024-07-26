import pandas as pd
from sklearn.model_selection import KFold
import os
import numpy as np

# 载入数据
df = pd.read_csv('HMU_ALL_PT_LABEL.csv')  # 请将 'STAD-209-50.csv' 替换为你的文件名

# 获取唯一的病人ID列表
patient_ids = df[df.columns[0]].unique()

# 随机打乱病人ID
np.random.shuffle(patient_ids)

# 创建splits文件夹，如果不存在
if not os.path.exists('splits'):
    os.makedirs('splits')

# 使用KFold进行五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 保存训练和验证集
for fold, (train_index, val_index) in enumerate(kf.split(patient_ids)):
    train_patient_ids = patient_ids[train_index]
    val_patient_ids = patient_ids[val_index]

    # 提取训练和验证集
    train_set = df[df[df.columns[1]].isin(train_patient_ids)]
    val_set = df[df[df.columns[1]].isin(val_patient_ids)]

    # 保存训练和验证集
    train_set.to_csv(f'splits/train_set_{fold}.csv', index=False)
    val_set.to_csv(f'splits/val_set_{fold}.csv', index=False)
