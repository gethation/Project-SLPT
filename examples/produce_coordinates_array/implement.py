import os
from datafactory import integration
import json
import os
from tqdm.auto import tqdm

schedule_path = 'schedule.json'
Data_path = 'Dataset'
input_video_path_list = [path for path in os.listdir(Data_path) if '.mp4' in path]

if not os.path.exists(schedule_path):
    with open(schedule_path, 'w') as f:
        json.dump({'video_path':input_video_path_list, 
                   'process': 0, 
                   'note':'100'}, f, indent=4)
    index = 0
    print('establish schedule')
else:
    with open(schedule_path, 'r') as f:
        x = json.load(f)
    input_video_path_list = x['video_path']
    index = x['process']




while True:

    with open(schedule_path, 'r') as f:
        x = json.load(f)
    input_video_path_list = x['video_path']
    index = x['process']
    if index+1 != len(input_video_path_list):
        with open(schedule_path, 'w') as f:
            json.dump({'video_path':input_video_path_list, 
                        'process': index+1, 
                        'note':'100'}, f, indent=4)
        
    video_path = input_video_path_list[index]
    print(video_path)
    integration(os.path.join(Data_path, video_path))

    if index+1 == len(input_video_path_list):
        break