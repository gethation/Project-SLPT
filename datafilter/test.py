import numpy as np
import json
import os
from utils import visualize
from tqdm import tqdm 

dataset_path = '2022-3-9-1400_segments'
output_folder = 'output'
std_list_path = 'standard_deviation.json'
std_list = {}
json_name = os.listdir(dataset_path)

for name in tqdm(json_name):

    with open(os.path.join(dataset_path, name),'r') as f:
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

    with open(os.path.join(output_folder, name),'w') as f:
        json.dump(selected_data.tolist(), f , indent=4)

    std_list[name] = standard_deviation_per_frame.tolist()


with open(std_list_path,'w') as f:
    json.dump(std_list, f , indent=4)    



