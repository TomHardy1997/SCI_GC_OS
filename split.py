import pandas as pd
from sklearn.model_selection import KFold
import os
import numpy as np

# 载入数据
df = pd.read_csv('HMU_ALL_PT_LABEL_13000filtered.csv')

# 获取唯一的病人ID列表（假设第一列是病人ID）
patient_ids = df[df.columns[0]].unique()

# 随机打乱病人ID
np.random.shuffle(patient_ids)

# 创建splits文件夹，如果不存在
if not os.path.exists('splits/filtered'):
    os.makedirs('splits/filtered')

# 使用KFold进行五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 保存训练和验证集
for fold, (train_index, val_index) in enumerate(kf.split(patient_ids)):
    train_patient_ids = patient_ids[train_index]
    val_patient_ids = patient_ids[val_index]

    # 提取训练和验证集
    train_set = df[df[df.columns[0]].isin(train_patient_ids)]
    val_set = df[df[df.columns[0]].isin(val_patient_ids)]
    
    # 对训练集的 censor 列进行交替
    train_set_censor = train_set['censor'].values
    train_set['censor'] = np.tile([0, 1], int(np.ceil(len(train_set_censor) / 2)))[:len(train_set_censor)]
    
    # 对验证集的 censor 列进行交替
    val_set_censor = val_set['censor'].values
    val_set['censor'] = np.tile([0, 1], int(np.ceil(len(val_set_censor) / 2)))[:len(val_set_censor)]

    # 保存训练和验证集
    train_set.to_csv(f'splits/filtered/train_set_{fold}.csv', index=False)
    val_set.to_csv(f'splits/filtered/val_set_{fold}.csv', index=False)

print("五折交叉验证的数据集已生成并保存至splits目录下。")
