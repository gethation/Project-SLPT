from utils import detect, procedure, fixed, extend, takeout_zero, split_video, parse_srt

import cv2
import mediapipe as mp
import numpy as np
import os
import json


def clip(input_video_path, base_filename):

    output_folder = fr'segment_temporary/{base_filename}_segments'
    os.makedirs(output_folder, exist_ok=True)


    srt_filename = fr'Dataset/{base_filename}.srt'
    subtitles = parse_srt(srt_filename)

    start_time = subtitles[0].get('start_seconds')
    end_time = subtitles[-1].get('start_seconds')
    print(start_time, end_time)
    split_video(input_video_path, output_folder, start_time, end_time)

def node_json(base_filename):

    input_dir = fr'segment_temporary/{base_filename}_segments'
    video_list = os.listdir(input_dir)

    output_folder = fr'segment_temporary/{base_filename}_segments'
    os.makedirs(output_folder, exist_ok=True)

    path = video_list[0]

    try:
        keypoint_coordinates = procedure(os.path.join(input_dir, path), crop=True, show=True)
        output_file = os.path.join(output_folder, os.path.splitext(path)[0].replace('_segment_0','')+'.json',)
        x = np.concatenate((np.array(keypoint_coordinates['hand']),
                        np.array(keypoint_coordinates['pose'])), 1)        
        
        with open(output_file, 'w') as f:
            json.dump(x.tolist(), f, indent=4)
    except:
        pass

def split_json(base_filename, time_span=60):

    uniform_length = 64

    input_folder = fr'segment_temporary/{base_filename}_segments'
    path = os.path.join(input_folder, base_filename +'_segment.json',)

    output_folder = fr'outcome/{base_filename}_segments'
    os.makedirs(output_folder, exist_ok=True)
    with open(path, 'r') as f:
        coordinates_jason = json.load(f)

    for pointer in range(0, len(coordinates_jason)-60, time_span//2):
        keypoint_coordinates = coordinates_jason[pointer:pointer+time_span]
        keypoint_coordinates = fixed(keypoint_coordinates)
        keypoint_coordinates = takeout_zero(keypoint_coordinates)
        keypoint_coordinates = extend(keypoint_coordinates, uniform_length).tolist()

        output_file = os.path.join(output_folder, f'{base_filename}'+'.json')

        with open(output_file, 'w') as f:
            json.dump(keypoint_coordinates, f, indent=4)

    
def integration(input_video_path):
    base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    clip(input_video_path, base_filename)
    node_json(base_filename)
    split_json(base_filename)
