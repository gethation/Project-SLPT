import json 
import os
from tqdm.auto import tqdm
from utils import web_cam, node_json, split_json, visualize, web_cam_photo
import gdown

datasetfolder = 'dataset'
word = 'gesture1'
ratio = 64 #每帧複製次數
interval = 4 #隔幾幀
length = 10


class_name = word
save_folder = os.path.join(datasetfolder, class_name)
input_video = os.path.join(save_folder, 'output.mp4')
input_json  = os.path.join(save_folder, 'nodes.json')
time_mark_path=os.path.join(save_folder, 'time mark.json')
output_folder = os.path.join(save_folder, 'output')
os.makedirs(save_folder, exist_ok=True)

import numpy as np

# web_cam(save_folder=save_folder, recording_time=length, break_time=2)
node_json(input_video, save_folder)
split_json(input_json,
           save_folder,
           time_mark_path,
           uniform_length=length*32)
json_path = os.path.join(save_folder,'segmented.json')
visualize(json_path, 64)


with open(json_path, 'r') as f:
    splited_json = json.load(f)
print(len(splited_json))
indices = [i for i in range(20)] + [(i+21) for i in range(20)]
data = np.array(splited_json)
selected_data = data[:, indices, :]

output_path = os.path.join(save_folder, class_name)
os.makedirs(output_path, exist_ok=True)
for i, index in enumerate(range(0, len(splited_json), 4)):
    output_container = []

    for _ in range(ratio):
        output_container.append(selected_data[index].tolist())

    with open(os.path.join(output_path, f'{word}_output_data_{i}.json'), 'w') as file:
        json.dump(output_container, file,indent=4)