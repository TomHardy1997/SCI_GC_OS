import torch
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

def load_and_check(file_path):
    try:
        print(f"Loading {file_path}")  # 调试信息，显示正在加载的文件
        x = torch.load(file_path)
        
        # 获取文件的实际维度
        actual_shape = x.shape
        
        return os.path.basename(file_path), actual_shape  # 返回文件名称和实际形状
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return os.path.basename(file_path), "Error loading"  # 如果加载失败，记录错误信息

# 获取文件路径列表
path = [os.path.join('/mnt/usb5/jijianxin/new_wsi/HMU_ALL_512_PT/', i) for i in os.listdir('/mnt/usb5/jijianxin/new_wsi/HMU_ALL_512_PT/')]

# 批量处理大小
batch_size = 100  # 每次处理100个文件

# 初始化输出文件
output_file = 'output_dimensions.csv'

# 如果输出文件已存在，先删除它
if os.path.exists(output_file):
    os.remove(output_file)

# 分批次处理文件并保存结果
for i in range(0, len(path), batch_size):
    batch_paths = path[i:i + batch_size]
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # 限制并行线程数为4
        results = list(tqdm(executor.map(load_and_check, batch_paths), total=len(batch_paths)))
    
    # 将文件名和形状转换为 DataFrame
    df = pd.DataFrame(results, columns=['filename', 'actual_shape'])
    
    # 追加保存到 CSV 文件
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

print(f"Processing complete. Results saved to {output_file}.")
