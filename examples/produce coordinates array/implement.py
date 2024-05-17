import os
from datafactory import integration
import json
import os
Data_path = 'Dataset'
input_video_path_list = os.listdir(Data_path)
for video_path in input_video_path_list:
    integration(os.path.join(Data_path, video_path))