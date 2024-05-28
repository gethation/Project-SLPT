import numpy as np
import json
import os
from utils import visualize
from tqdm import tqdm 

dataset_path = r'D:\test_outcome'
output_folder = r'D:\test_dataset'
std_list_path = 'standard_deviation.json'
std_list = {}
os.makedirs(output_folder, exist_ok=True)

def get_files_in_folder(folder_path):
    files = []
    # 使用os.walk遍历目录及其子目录
    for root, _, filenames in os.walk(folder_path):
        # 遍历当前目录中的文件
        for filename in filenames:
            # 拼接文件路径
            files.append((root, filename))
    return files

json_path = get_files_in_folder(dataset_path)


for (root, filename) in tqdm(json_path):

    with open(os.path.join(root, filename),'r') as f:
        x = json.load(f)
    data = np.array(x)

    #保留的點的索引
    data = np.array(x)
    indices = connections = [i for i in range(20)] + [(i+21) for i in range(20)] + [(i+42) for i in [0,11,12,13,14]]
    selected_data = data[:, indices, :]

    differences = np.diff(data, axis=0)
    distances = np.linalg.norm(differences, axis=2)

    # 計算每一幀的點距離變化的標準差
    standard_deviation_per_frame = np.std(distances, axis=1)

    with open(os.path.join(output_folder, filename),'w') as f:
        json.dump(selected_data.tolist(), f , indent=4)

    std_list[filename] = standard_deviation_per_frame.tolist()


with open(std_list_path,'w') as f:
    json.dump(std_list, f , indent=4)    



