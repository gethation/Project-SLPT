from utils import detect, procedure, fixed, extend, takeout_zero, parse_srt

import cv2
import mediapipe as mp
import numpy as np
import os
import json

def node_json(input_video_path, base_filename):

    output_folder = fr'segment_temporary/{base_filename}_segments'
    os.makedirs(output_folder, exist_ok=True)

    srt_filename = fr'Dataset/{base_filename}.srt'
    subtitles = parse_srt(srt_filename)

    start_time = subtitles[0].get('start_seconds')
    end_time = subtitles[-1].get('start_seconds')
    try:
        keypoint_coordinates = procedure(input_video_path,
                                         crop=True,
                                         start_time=start_time,
                                         end_time=end_time,
                                         show=True)
        output_file = os.path.join(output_folder, base_filename +'_segment.json',)
        x = np.concatenate((np.array(keypoint_coordinates['hand']),
                        np.array(keypoint_coordinates['pose'])), 1)        
        
        with open(output_file, 'w') as f:
            json.dump(x.tolist(), f, indent=4)
    except:
        pass

def split_json(base_filename, time_span=60):

    uniform_length = time_span

    input_folder = fr'segment_temporary/{base_filename}_segments'
    path = os.path.join(input_folder, base_filename +'_segment.json',)

    output_folder = fr'outcome/{base_filename}_segments'
    os.makedirs(output_folder, exist_ok=True)
    with open(path, 'r') as f:
        coordinates_jason = json.load(f)

    for i, pointer in enumerate(range(0, len(coordinates_jason)-time_span, time_span//2)):
        keypoint_coordinates = coordinates_jason[pointer:pointer+time_span]
        keypoint_coordinates = fixed(keypoint_coordinates)
        keypoint_coordinates = takeout_zero(keypoint_coordinates)
        keypoint_coordinates = extend(keypoint_coordinates, uniform_length).tolist()

        output_file = os.path.join(output_folder, f'{base_filename}_{i}'+'.json')

        with open(output_file, 'w') as f:
            json.dump(keypoint_coordinates, f, indent=4)

    
def integration(input_video_path):
    base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    node_json(input_video_path, base_filename)
    split_json(base_filename, 120)
