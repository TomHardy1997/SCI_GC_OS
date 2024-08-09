import torch
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

def load_and_check(file_path):
    try:
        print(f"Loading {file_path}")  # 调试信息，显示正在加载的文件
        x = torch.load(file_path)
        # 检查 shape[0] 是否小于 12000
        if x.shape[0] < 13000:
            return os.path.basename(file_path)  # 返回文件名称而不是张量
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

# 获取文件路径列表
path = [os.path.join('/mnt/usb5/jijianxin/new_wsi/HMU_ALL_512_PT/', i) for i in os.listdir('/mnt/usb5/jijianxin/new_wsi/HMU_ALL_512_PT/')]

# 初始化输出文件
output_file = 'output_13000.csv'

# 如果输出文件已存在，先删除它
if os.path.exists(output_file):
    os.remove(output_file)

# 使用线程池并行处理所有文件
with ThreadPoolExecutor(max_workers=4) as executor:  # 限制并行线程数为4
    results = list(tqdm(executor.map(load_and_check, path), total=len(path)))

# 过滤掉 None 结果，并只保存文件名
valid_files = [result for result in results if result is not None]

# 将文件名转换为 DataFrame
df = pd.DataFrame(valid_files, columns=['filename'])

# 保存到 CSV 文件
df.to_csv(output_file, index=False)

print(f"Processing complete. Results saved to {output_file}.")
