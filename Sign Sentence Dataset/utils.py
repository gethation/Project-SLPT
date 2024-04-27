import os
import opencc
import numpy as np
import mediapipe as mp
cc = opencc.OpenCC('hk2s')
from moviepy.video.io.VideoFileClip import VideoFileClip
import jieba
import json
import cv2
from tqdm.auto import tqdm
from os import path as osp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # 设置最大追踪手的数量
mp_drawing = mp.solutions.drawing_utils

def time_to_seconds(time_str):
    time_parts = time_str.replace(',', ':').split(':')
    h = int(time_parts[0])
    m = int(time_parts[1])
    s = int(time_parts[2])
    ms = int(time_parts[3])
    
    return h * 3600 + m * 60 + s + ms / 1000

def format_srt(input_file):
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        content = file.readlines()

    formatted_content = ['\n'] + content
    
    return formatted_content

def parse_srt(filename):

    lines = format_srt(filename)
    subtitles = []

    subtitle = None
    for line in lines:
        line = line.strip()
        if line.isdigit():
            if subtitle is not None:
                subtitles.append(subtitle)
            subtitle = {'index': int(line)}
        elif '-->' in line:
            start_end = line.split('-->')
            start_time, end_time = [time.strip() for time in start_end]
            if subtitle is not None:
                subtitle['start_time'] = start_time
                subtitle['end_time'] = end_time
                subtitle['start_seconds'] = time_to_seconds(start_time)
                subtitle['end_seconds'] = time_to_seconds(end_time)
        elif line and subtitle is not None:
            if 'text' in subtitle:
                subtitle['text'] += ' ' + line
            else:
                subtitle['text'] = line
    if subtitle is not None:
        subtitles.append(subtitle)

    return subtitles

def make_context(srt):

    context_list = []
    filter_text = ['好','那']
    for x in srt:
        try:
            _ = x['text']
            switch = True
        except:
            switch = False
        if switch:
            word_time_position = np.round(np.linspace(x['start_seconds'],
                                                        x['end_seconds'],
                                                        len(x['text'])), 2)
            l_point = 0
            for sentence in x['text'].split():
                if sentence not in filter_text:
                    context_list.append({'sentence':sentence,
                                            'start_seconds':word_time_position[l_point],
                                            'end_seconds':word_time_position[l_point+len(sentence)-1]})
                    l_point += len(sentence)+1
                    
    return context_list

def reconstruct(context_list):

    improved_context = []
    concated_sentence= ''
    end_seconds = -1
    start_seconds = -1
    for i, x in enumerate(context_list):
        sentence = x['sentence']
        splited_sentence = [i for i in jieba.cut(cc.convert(sentence), cut_all=False, HMM=True)]
        filtered_list = [item for item in splited_sentence if not item.isdigit()]
        # print(filtered_list, x['sentence'])
        if len(filtered_list) < 4:
            concated_sentence += sentence

        else:
            end_seconds = context_list[i-1]['end_seconds']
            improved_context.append({'sentence':concated_sentence,
                                    'start_seconds':start_seconds,
                                    'end_seconds':end_seconds})
            concated_sentence = sentence
            start_seconds = x['start_seconds']

    for i, x in enumerate(improved_context):
        if x['sentence']=='':
            improved_context.pop(i)

    return improved_context

def detect(frame, keypoint_coordinates, show):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    landmark_coords = [[0,0] for _ in range(42)]
    if results.multi_hand_landmarks:
        for (landmarks, handedness) in zip(results.multi_hand_landmarks, results.multi_handedness):
            if show:
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS)

            replacement = []
            for i, landmark in enumerate(landmarks.landmark):
                landmark_x = int(landmark.x * frame.shape[1])
                landmark_y = int(landmark.y * frame.shape[0])
                replacement.append([landmark_x, landmark_y])
            if handedness.classification[0].index == 1: #index 1 = Right
                landmark_coords[:21] = replacement
            else:
                landmark_coords[21:] = replacement
    keypoint_coordinates.append(landmark_coords)
    return frame

# def procedure(video_path, crop, show=False):
#     cap = cv2.VideoCapture(video_path)

#     keypoint_coordinates = []
    
#     # 获取视频总帧数
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # 使用tqdm创建进度条
#     with tqdm(total=total_frames, desc="Processing Frames") as pbar:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = pre_adjust(frame, crop)
#             detect(frame, keypoint_coordinates, show)

#             if show:
#                 cv2.imshow('Hand Tracking', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
            
#             # 更新进度条
#             pbar.update(1)

#     if show:
#         cap.release()
#         cv2.destroyAllWindows()

#     return keypoint_coordinates

def procedure(video_path, crop, start_frame, end_frame, show=False):
    cap = cv2.VideoCapture(video_path)

    keypoint_coordinates = []

    # 设置视频的当前帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 获取视频总帧数
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), end_frame)

    # 使用tqdm创建进度条
    with tqdm(total=total_frames - start_frame, desc="Processing Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            
            frame = pre_adjust(frame, crop)
            detect(frame, keypoint_coordinates, show)

            if show:
                cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 更新进度条
            pbar.update(1)

    if show:
        cap.release()
        cv2.destroyAllWindows()

    return keypoint_coordinates

# 例如，从第10帧到第100帧的处理：
# start_frame = 10
# end_frame = 100
# result = procedure("your_video_path.mp4", your_crop_parameter, start_frame, end_frame, show=True)

def pre_adjust(frame, crop):
    if crop:
        frame = frame[360:750, 1460:1810]
        # height, width, _ = frame.shape
        # box_size = 400  
        # x = (width - box_size) // 2
        # y = (height - box_size) // 2
        # frame = frame[y:y+box_size, x:x + box_size]
    if frame.shape[1] >= frame.shape[0]:
        l = frame.shape[1]//2-frame.shape[0]//2
        r = frame.shape[1]//2+frame.shape[0]//2
        frame = frame[:,l:r,:]
    if frame.shape[1] <= frame.shape[0]:
        t = frame.shape[0]//2-frame.shape[1]//2
        b = frame.shape[0]//2+frame.shape[1]//2
        frame = frame[t:b,:,:]
    frame = cv2.resize(frame, (640, 640))

    return frame

def fixed(keypoint_coordinates):
    keypoint_coordinates = np.array(keypoint_coordinates)
    f = 0
    l_index = -1
    r_index = -1
    print(keypoint_coordinates.shape)
    for i in range(keypoint_coordinates[:,f].shape[0]-1):
        left = keypoint_coordinates[i,f]
        right = keypoint_coordinates[i+1,f]
        if sum(left) != 0 and sum(right) == 0:
            l_index = i+1
        if sum(left) == 0 and sum(right) != 0:
            r_index = i+1
        if l_index < r_index:
            keypoint_coordinates[l_index-1:r_index+1,0:20] = np.linspace(keypoint_coordinates[l_index-1,0:20],
                                                                        keypoint_coordinates[r_index,0:20],
                                                                        keypoint_coordinates[l_index-1:r_index+1,f].shape[0])
    f = 21
    l_index = -1
    r_index = -1
    for i in range(keypoint_coordinates[:,f].shape[0]-1):
        left = keypoint_coordinates[i,f]
        right = keypoint_coordinates[i+1,f]
        if sum(left) != 0 and sum(right) == 0:
            l_index = i+1
        if sum(left) == 0 and sum(right) != 0:
            r_index = i+1
        if l_index < r_index:
            keypoint_coordinates[l_index-1:r_index+1,21:] = np.linspace(keypoint_coordinates[l_index-1,21:],
                                                                        keypoint_coordinates[r_index,21:],
                                                                        keypoint_coordinates[l_index-1:r_index+1,f].shape[0])
    return keypoint_coordinates

def elongate(keypoint_coordinates_dim, new_length):

    new_indices = np.linspace(0, len(keypoint_coordinates_dim) - 1, new_length)
    new_array = np.interp(new_indices, np.arange(len(keypoint_coordinates_dim)), keypoint_coordinates_dim)

    return new_array

def extend(keypoint_coordinates, new_length):
    filled_keypoint_coordinates = np.zeros((new_length, 42, 2))
    for i in range(keypoint_coordinates.shape[1]-1):
        filled_keypoint_coordinates[:, i, 0] = elongate(keypoint_coordinates[:, i, 0], new_length)
        filled_keypoint_coordinates[:, i, 1] = elongate(keypoint_coordinates[:, i, 1], new_length)

    return filled_keypoint_coordinates

def takeout_zero(keypoint_coordinates):
    zero_cols = np.all(keypoint_coordinates == 0, axis=(1,2))

    # 刪除全為零的列
    arr_without_zero_cols = keypoint_coordinates[~zero_cols, :, :]
    return arr_without_zero_cols

#todo file management
def clip_video(video_path, output_path, start_time, limit_time):
    video_clip = VideoFileClip(video_path)
    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    segment_clip = video_clip.subclip(start_time, limit_time)
    segment_clip.write_videofile(output_path)

    video_clip.reader.close()

def mk_json(str_path, save_path):
    srt = parse_srt(str_path)
    context_list = make_context(srt)
    improved_context = reconstruct(context_list)

    with open(save_path, 'w') as json_file:
        json.dump(improved_context, json_file, ensure_ascii=False)

def clip(input_video_path, input_chart_path, output_path):

    with open(input_chart_path, 'r', encoding='utf-8') as json_file:
        chart = json.load(json_file)

    start_time = chart[0]['start_seconds']
    end_time = chart[-1]['end_seconds']

    clip_video(input_video_path, output_path, start_time, end_time)

#todo file management
def node_json(video_path, output_path, input_chart_path):

    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(input_chart_path, 'r', encoding='utf-8') as json_file:
        chart = json.load(json_file)

    start_time = chart[0]['start_seconds']
    end_time = chart[-1]['end_seconds']

    keypoint_coordinates = procedure(video_path, True, int(start_time*fps), int(end_time*fps))
                            
    with open(output_path, 'w') as f:
        json.dump(keypoint_coordinates, f, indent=4)

def split_json(input_json_name, input_chart_path, output_folder, basename , video_path):

    with open(input_json_name, 'r') as f:
        coordinates_jason = json.load(f)

    with open(input_chart_path, 'r', encoding='utf-8') as json_file:
        chart = json.load(json_file)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # target_frame = int(517.17 * fps)
    benchmark = chart[0]['start_seconds']
    offset = 1.8
    for i, item in enumerate(chart):
        try:
            offset_length = round((chart[i]['end_seconds']-chart[i]['start_seconds'])*0.08, 3)
            keypoint_coordinates = coordinates_jason[int((item['start_seconds']-benchmark+offset-offset_length) * fps):
                                                     int((item['end_seconds']-benchmark+offset+offset_length) * fps)]
            keypoint_coordinates = fixed(keypoint_coordinates)
            keypoint_coordinates = takeout_zero(keypoint_coordinates).tolist()
            # keypoint_coordinates = extend(keypoint_coordinates, uniform_length).tolist()

            output_path = os.path.join(output_folder, basename+f'pair_{i}')
            with open(output_path, 'w') as f:
                json.dump({'sentence':item['sentence'],'coordinates':keypoint_coordinates}, f, ensure_ascii=False)
        except:
            pass

def compose(video_path):
    basename = osp.splitext(osp.basename(video_path))[0]
    data_folder = 'dataset'
    segment_folder = osp.join('segment_temporary', basename)

    os.makedirs(segment_folder, exist_ok=True)
    mk_json(osp.join(data_folder, basename+'.srt'),
            osp.join(segment_folder, basename+'.json'))
    
    # clip(video_path,
    #      osp.join(segment_folder, basename+'.json'),
    #      osp.join(segment_folder, basename+'.mp4'))
    
    node_json(video_path,
              osp.join(segment_folder, basename+'-node.json'),
              osp.join(segment_folder, basename+'.json'))
    
    os.makedirs(osp.join(segment_folder,basename+'pair'), exist_ok=True)

    split_json(osp.join(segment_folder, basename+'-node.json'),
               osp.join(segment_folder, basename+'.json'),
               osp.join(segment_folder,basename+'pair'),
               basename,
               video_path)