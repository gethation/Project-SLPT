import json 
import os
from tqdm.auto import tqdm
from utils import web_cam, node_json, split_json, visualize, web_cam_photo
import gdown

datasetfolder = 'dataset'
word = '_'
class_name = word
save_folder = os.path.join(datasetfolder, class_name)
input_video = os.path.join(save_folder, 'output.mp4')
input_json  = os.path.join(save_folder, 'nodes.json')
time_mark_path=os.path.join(save_folder, 'time mark.json')
output_folder = os.path.join(save_folder, 'output')
os.makedirs(save_folder, exist_ok=True)

web_cam(save_folder=save_folder, recording_time=4, break_time=2)