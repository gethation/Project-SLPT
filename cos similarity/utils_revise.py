import cv2
import mediapipe as mp
import numpy as np
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchvision.transforms as transforms
import json
import time
import torchvision
import torch
from torch import nn
from model import MAE
from utils_cam import web_cam, node_json, split_json, visualize
from tqdm.auto import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # 设置最大追踪手的数量
mp_drawing = mp.solutions.drawing_utils

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

def procedure(video_path, crop, box=  None, show = False):
    cap = cv2.VideoCapture(video_path)

    keypoint_coordinates = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        i+=1
        if not ret:
            break
        frame = pre_adjust(frame, crop)
        detect(frame, keypoint_coordinates, show)

        if show:
            cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if show:
        cap.release()
        cv2.destroyAllWindows()
    
    return keypoint_coordinates, i

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

def pre_adjust(frame, crop):
    if crop:
        # frame = frame[360:750, 1460:1810]
        height, width, _ = frame.shape
        box_size = 520
        x = (width - box_size) // 2
        y = (height - box_size) // 2
        frame = frame[y:y+box_size, x:x + box_size]
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

def split_video(video_path, output_folder, start_time, limit_time):
    video_clip = VideoFileClip(video_path)
    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    segment_clip = video_clip.subclip(start_time, limit_time)
    segment_filename = os.path.join(output_folder, f"{base_filename}_segment_{0}.mp4")
    segment_clip.write_videofile(segment_filename)

    video_clip.reader.close()

def time_to_seconds(time_str):
    time_parts = time_str.replace(',', ':').split(':')
    h = int(time_parts[0])
    m = int(time_parts[1])
    s = int(time_parts[2])
    ms = int(time_parts[3])
    
    return h * 3600 + m * 60 + s + ms / 1000

def parse_srt(filename):
    subtitles = []

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

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

# def rotate_point(point, center, angle):

#     offset = point - center

#     # 创建旋转矩阵
#     rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
#                                 [np.sin(angle), np.cos(angle)]])

#     # 使用旋转矩阵旋转偏移量
#     rotated_offset = np.dot(rotation_matrix, offset)

#     # 将旋转后的偏移量添加回圆心坐标以获取旋转后的点
#     rotated_point = rotated_offset + center

#     return rotated_point

# def scaling_point(point, center, scale_factor):

#     # 计算点相对于圆心的偏移量
#     offset = point - center

#     # 使用缩放因子对偏移量进行缩放
#     scaled_offset = offset * scale_factor

#     # 将缩放后的偏移量添加回圆心坐标以获取缩放后的点
#     scaled_point = scaled_offset + center

#     return scaled_point

# def random_rotate(coordinates):
#     angle = np.radians(np.random.randint(-20, 20))
#     center = [320, 320]
#     rotated_coordinates = np.empty_like(coordinates)
#     for i, frame in enumerate(coordinates):
#         for j, point in enumerate(frame):
#             rotated_coordinates[i,j] = rotate_point(point, center, angle)
#     return rotated_coordinates

# def random_scaling(coordinates):
#     scale_factor = np.random.uniform(0.6, 1.5)
#     center = [320, 320]
#     scaled_coordinates = np.empty_like(coordinates)
#     for i, frame in enumerate(coordinates):
#         for j, point in enumerate(frame):
#             scaled_coordinates[i,j] = scaling_point(point, center, scale_factor)
#     return scaled_coordinates

# def offsetalize(coordinates):
#     offset = np.random.randint(-30, 30, 2)
#     offsetalized_coordinates = np.empty_like(coordinates)
#     for i, frame in enumerate(coordinates):
#         for j, point in enumerate(frame):
#             offsetalized_coordinates[i,j] = point + offset
#     return offsetalized_coordinates

x = [(i, i+1) for i in range(0,4)]+[(i, i+1) for i in range(5,8)]+[(i, i+1) for i in range(9,12)]+[(i, i+1) for i in range(13,16)]+[(i, i+1) for i in range(17,19)]+[(0,5),(0,17),(5,9),(9,13),(13,17)]
connections = [i for i in x]+[ (i[0]+21, i[1]+21) for i in x]

# def visualize(input_file, video_path=None):
#     with open(input_file, 'r') as f:
#         coordinates = json.load(f)
#     coordinates = np.array(coordinates)
#     # coordinates = takeout_zero(coordinates)
#     # coordinates = random_rotate(coordinates)
#     # coordinates = random_scaling(coordinates)
#     # coordinates = offsetalize(coordinates)
#     keypoint_coordinates = extend(coordinates, 64).astype(np.int16)
#     # print(coordinates.shape, keypoint_coordinates.shape)
#     # 创建黑色背景
#     background = 255 * np.ones((640, 640, 3), dtype=np.uint8)
#     if video_path != None:
#         cap = cv2.VideoCapture(video_path)

#     # 指定要连接的点的索引

#     for keypoints in keypoint_coordinates:
#         # 创建黑色背景的副本
#         if video_path != None:
#             ret, frame = cap.read()
#             frame = pre_adjust(frame, True)
#         else:
#             frame = background.copy()

#         # 绘制关键点
#         for i, keypoint in enumerate(keypoints):
#             x, y = keypoint[0], keypoint[1]
#             if i >= 21:
#                 color = (0, 0, 255)
#             else:
#                 color = (255, 0, 0)
#             if i == 41 or i == 20:
#                 pass
#             else:
#                 cv2.circle(frame, (x, y), 5, color, -1)

#         # 连接指定的点
#         for connection in connections:
#             start_point = tuple(keypoints[connection[0]])
#             end_point = tuple(keypoints[connection[1]])
#             cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

#         cv2.imshow('Hand Key Points', frame)

#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()



def put_text(frame, elapsed_time, num=0):

    cv2.putText(frame, f'Time: {float(elapsed_time):.1f} seconds', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(frame, f'NuM: {num}/{100}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 计算方框的位置
    height, width, _ = frame.shape
    box_size = 360
    x = (width - box_size) // 2
    y = (height - box_size) // 2

    # 绘制方框
    cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 0, 0), 1)
    
    if elapsed_time <= 1:
        cv2.rectangle(frame, (0, height-15), (int((elapsed_time/1)*width), height), (0, 0, 250), -1)
    if elapsed_time > 1:
        cv2.rectangle(frame, (0, height-15), (int(((elapsed_time-1)/2)*width), height), (0, 250, 0), -1)

    return frame

# def web_cam(break_time=1, save_folder = ''):
#     segment = []
#     cap = cv2.VideoCapture(0)

#     # 检查摄像头是否成功打开
#     if not cap.isOpened():
#         print("无法打开摄像头")
#         exit()

#     # 定义视频编码器并创建VideoWriter对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
#     out = cv2.VideoWriter(os.path.join(save_folder, 'output.mp4'), fourcc, 30.0, (640, 480))  # 参数分别为输出文件名，编码器，帧率和分辨率


#     # 初始化实时帧率计算器
#     frame_counter = 0
#     start_frame = 0
#     end_frame = 0
#     num = 0
#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()  # 读取一帧图像
#         # frame = cv2.resize(frame, (1280, 960))
#         if not ret:
#             print("无法获取图像")
#             break
#         # 将当前帧写入输出视频文件
#         out.write(frame)
#         frame_counter += 1
#         elapsed_time = time.time() - start_time


#         # 计算经过的时间
#         frame = put_text(frame, elapsed_time, num)


#         if elapsed_time < break_time:
#             start_frame = frame_counter

#         if elapsed_time >= break_time+2:
#             start_time = time.time()

#             end_frame = frame_counter
#             segment.append((start_frame, end_frame))
#             num += 1

#         # 在窗口中显示当前帧
#         frame = cv2.resize(frame, (1280, 960))
#         cv2.imshow('', frame)
#         # 按下 'q' 键退出循环
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


#     # 释放所有资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     with open(os.path.join(save_folder,'time mark.json'), 'w') as file:
#         json.dump(segment, file)

# def node_json(input_video, output_folder):
#     keypoint_coordinates,i = procedure(input_video, crop=True)
#     print(len(keypoint_coordinates), i)
#     output_file = os.path.join(output_folder, 'nodes.json')
    
#     with open(output_file, 'w') as f:
#         json.dump(keypoint_coordinates, f, indent=4)

# def split_json(input_json, output_folder, time_mark_path):

#     uniform_length = 64

#     os.makedirs(output_folder, exist_ok=True)

#     with open(input_json, 'r') as f:
#         coordinates_jason = json.load(f)
#     with open(time_mark_path, 'r') as f:
#         time_mark = json.load(f)
    
#     for i, (pointerL, pointerR) in enumerate(time_mark):
#         try:
#             keypoint_coordinates = coordinates_jason[pointerL:pointerR]
#             keypoint_coordinates = fixed(keypoint_coordinates)
#             keypoint_coordinates = takeout_zero(keypoint_coordinates)
#             keypoint_coordinates = extend(keypoint_coordinates, uniform_length).tolist()

#             output_file = os.path.join(output_folder, f'segment_{len(os.listdir(output_folder))}'+'.json',)

#             with open(output_file, 'w') as f:
#                 json.dump(keypoint_coordinates, f, indent=4)
#         except:
#             pass


class ViT(nn.Module):
    def __init__(self, pretrained_model):
        super(ViT, self).__init__()
        self.backbone = pretrained_model.backbone
        self.out_dim = 80

    def forward(self, images):
        batch_size = images.shape[0]
        images = images.reshape((batch_size, 64, self.out_dim))
        x = self.backbone(images)
        return x
    
def buit_eval_model(backbone_path = ''):
    vit = torchvision.models.vit_b_16(weights=None)
    pretrained_model = MAE(vit, 64, 80)
    pretrained_model.load_state_dict(torch.load(backbone_path))

    model = ViT(pretrained_model)

    model = model.to('cuda')
    return model.eval()

def coordinate_transform(coordinate):
    coordinate = np.array(coordinate)
    container = np.empty((64, 40, 2))
    container[:, :20, :] = coordinate[:, :20, :]
    container[:, 20:, :] = coordinate[:, 21:41, :]
    return container

def normalize_data(tensor, mean, std):
    # mean = tensor.mean()
    # std = tensor.std()
    normalize = transforms.Normalize(mean=[mean], std=[std])
    normalized_tensor = normalize(tensor)
    return normalized_tensor

def load_json(path):# revise
    with open(path, 'r') as file:
        json_ = json.load(file)
    x = coordinate_transform(json_)
    return x

import os

def clear_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            # 删除文件
            os.remove(file_path)
        for dir in dirs:
            # 构建子文件夹的完整路径
            dir_path = os.path.join(root, dir)
            # 删除子文件夹
            os.rmdir(dir_path)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

def inverse_euclidean_distance(vector1, vector2):
    distance = np.sqrt(np.sum((vector1 - vector2) ** 2))
    return 1 / (1 + distance)

def pearson_correlation_coefficient(vector1, vector2):
    # 计算每个向量的均值
    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)

    # 计算协方差和各自的标准差
    covariance = np.sum((vector1 - mean1) * (vector2 - mean2))
    standard_deviation1 = np.sqrt(np.sum((vector1 - mean1) ** 2))
    standard_deviation2 = np.sqrt(np.sum((vector2 - mean2) ** 2))

    # 计算皮尔森相关系数
    correlation = covariance / (standard_deviation1 * standard_deviation2)
    return correlation


def video_clip_procedure(input_path, output_path, class_name):
    video_capture = cv2.VideoCapture(input_path) # 替换为你的视频文件路径

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # 读取所有视频帧并存储在 frames 数组中
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)

    # 将 frames 转换为 NumPy 数组
    frames_array = np.array(frames)

    # 创建一个窗口
    cv2.namedWindow(f'{class_name}')

    # 创建一个滑块
    cv2.createTrackbar('Frame', f'{class_name}', 0, len(frames_array)-1, lambda x: None)
    point = []

    while True:
        # 获取滑块的当前值
        current_frame = cv2.getTrackbarPos('Frame', f'{class_name}')

        # 显示当前帧
        if len(point)==1 and current_frame<=point[0]:
            cv2.imshow(f'{class_name}', frames_array[point[0]])
        else:
            cv2.imshow(f'{class_name}', frames_array[current_frame])
        key = cv2.waitKey(25) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):  # Press 'r' to record the current frame
            print(f"Selected Frame: {current_frame}")
            point.append(current_frame)
            if len(point)==2:
                break
    cv2.destroyAllWindows()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要选择适当的编解码器
    output_video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)  # 替换为输出视频文件路径
    try:
        for f in frames[point[0]:point[1]+1]:
            output_video.write(f)
        output_video.release()
        video_capture.release()

        print(f"Video saved successfully from frame {point[0]} to frame {point[1]}.")
    except:
        print('ERROR')

def video_clip_stream(video_dir, index):

    video_path_list = [os.path.join(video_dir,i) for i in os.listdir(video_dir)]
    class_name_list = [os.path.basename(i).replace('.mp4', '') for i in video_path_list]
    class_name = class_name_list[index]

    clear_folder(rf'dataset\{class_name}\{class_name}')
    save_folder = os.path.join('dataset', class_name)
    os.makedirs(save_folder, exist_ok=True)
    video_path, output_path = video_path_list[index], rf'dataset\{class_name}\output.mp4'
    video_clip_procedure(video_path, output_path, class_name)

# def video_process_stream(model, video_dir, index, mean, std):

#     video_path_list = [os.path.join(video_dir,i) for i in os.listdir(video_dir)]
#     class_name_list = [os.path.basename(i).replace('.mp4', '') for i in video_path_list]
#     class_name = class_name_list[index]

#     save_folder = os.path.join('dataset', class_name)
#     input_video = os.path.join(save_folder, 'output.mp4')
#     input_json  = os.path.join(save_folder, 'nodes.json')
#     time_mark_path=os.path.join(save_folder, 'time mark.json')
#     output_folder = os.path.join(save_folder, class_name)
    
#     video_path, output_path = video_path_list[index], rf'dataset\{class_name}\output.mp4'
#     video_path = output_path

#     node_json(video_path, save_folder, show=True)
#     split_json(input_json, output_folder)
#     json_path = rf'dataset\{class_name}\{class_name}\segment.json'
#     visualize(json_path)
#     x = load_json(json_path, mean, std).to('cuda')
#     prediction = model(x)

#     return prediction, class_name

def list_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders

def video_process_stream_class(model, x, mean, std, augment, i=0):# revise

    # class_name_list =  list_folders(dataset_dir)
    # class_name = class_name_list[index]

    # json_path = rf'{dataset_dir}\{class_name}\{class_name}\segment_{i}.json'
    # # visualize(json_path)
    # x = load_json(json_path)


    if augment:
        x = augmentation(x)
    data = torch.as_tensor(torch.from_numpy(x), dtype=torch.float32)
    batch = normalize_data(data, mean, std)
    batch = batch.unsqueeze(0)
    batch = batch.to('cuda')


    prediction = model(batch)

    return prediction
def load_json_expriment(path, mean, std, augment = False):
    with open(path, 'r') as file:
        json_ = json.load(file)
    x = coordinate_transform(json_)
    if augment:
        x = augmentation(x)
    data = torch.as_tensor(torch.from_numpy(x), dtype=torch.float32)
    batch = normalize_data(data, mean, std)

    return batch.unsqueeze(0)
def video_process_stream_expriment(model, dataset_dir, index, mean, std, augment, i=0):

    class_name_list =  list_folders(dataset_dir)
    class_name = class_name_list[index]

    json_path = rf'{dataset_dir}\{class_name}\{class_name}\segment_{i}.json'
    # visualize(json_path)
    x = load_json_expriment(json_path, mean, std, augment=augment).to('cuda')
    prediction = model(x)

    return prediction, class_name

def rotate_point(point, center, angle):

    offset = point - center

    # 创建旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # 使用旋转矩阵旋转偏移量
    rotated_offset = np.dot(rotation_matrix, offset)

    # 将旋转后的偏移量添加回圆心坐标以获取旋转后的点
    rotated_point = rotated_offset + center

    return rotated_point

def scaling_point(point, center, scale_factor):

    # 计算点相对于圆心的偏移量
    offset = point - center

    # 使用缩放因子对偏移量进行缩放
    scaled_offset = offset * scale_factor

    # 将缩放后的偏移量添加回圆心坐标以获取缩放后的点
    scaled_point = scaled_offset + center

    return scaled_point

def random_rotate(coordinates):
    angle = np.radians(np.random.randint(-5, 5))
    center = [320, 320]
    rotated_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            rotated_coordinates[i,j] = rotate_point(point, center, angle)
    return rotated_coordinates

def random_scaling(coordinates):
    scale_factor = np.random.uniform(0.8, 1.2)
    center = [320, 320]
    scaled_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            scaled_coordinates[i,j] = scaling_point(point, center, scale_factor)
    return scaled_coordinates

def offsetalize(coordinates):
    offset = np.random.randint(-30, 30, 2)
    offsetalized_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            offsetalized_coordinates[i,j] = point + offset
    return offsetalized_coordinates
def vertical_line():
    x = np.random.randint(280, 360)  # 隨機生成垂直線的 x 座標
    return x

def flip_point_vertical(point, x):
    x_old, y_old = point
    x_new = 2*x - x_old
    return x_new, y_old

def flip(coordinates):
    coordinates_flipped = np.copy(coordinates)

    x_vertical = vertical_line()

    for i in range(coordinates.shape[0]):
        for j in range(coordinates.shape[1]):
            coordinates_flipped[i, j] = flip_point_vertical(coordinates[i, j], x_vertical)
    return coordinates_flipped

def augmentation(x):
    # coordinates = np.array(x)
    # if np.random.rand()>0.5:
    #     coordinates = flip(coordinates)
    coordinates = random_rotate(x)
    coordinates = random_scaling(x)
    keypoint_coordinates = offsetalize(coordinates)
    return keypoint_coordinates

def sort(array1, array2):
    pair = []
    paired_arrays = list(zip(array1, array2))
    sorted_arrays = sorted(paired_arrays, key=lambda x: x[0], reverse=True)
    sorted_array1, sorted_array2 = zip(*sorted_arrays)

    for item in zip(sorted_array1, sorted_array2):
        pair.append(item)
    return pair

# def calculate_cosine_similarity(vector1, feature_path = 'feature.json'):
#     with open(feature_path, 'r') as f:
#         class_features = json.load(f)
#     id_vector = [value for value in class_features.items()]
#     similarity_list, id_list = [], []
#     for id, vector2 in id_vector:
#         similarity_list.append(round(cosine_similarity(vector1=np.array(vector1), vector2=np.array(vector2)), 4))
#         id_list.append(id)
#         # print(round(cosine_similarity(vector1=np.array(vector1), vector2=np.array(vector2)), 4), id)
#     return sort(similarity_list, id_list)

# def calculate_cosine_similarity(vector, feature_path = 'feature.json'):
#     with open(feature_path, 'r') as f:
#         class_features = json.load(f)
#     id_vector = [value for value in class_features.items()]
#     similarity_list, id_list = [], []
#     for id, vector_i in tqdm(id_vector):
#         # print(id, len(vector_i))
#         container = []
#         for feature in vector:
#             for feature_i in vector_i:
#                 container.append(cosine_similarity(vector1=np.array(feature), vector2=np.array(feature_i)))
#         # np.mean(np.array(container))
#         similarity_list.append(round(np.mean(np.array(container)), 4))
#         id_list.append(id)
#     return sort(similarity_list, id_list)

def judge(model, mean, std, sample_num = 100, visualized = False):

    class_name = 'record'
    clear_folder(rf'dataset\{class_name}\{class_name}')
    save_folder = os.path.join('dataset', class_name)
    input_video = os.path.join(save_folder, 'output.mp4')
    input_json  = os.path.join(save_folder, 'nodes.json')
    time_mark_path=os.path.join(save_folder, 'time mark.json')
    output_folder = os.path.join(save_folder, class_name)
    os.makedirs(save_folder, exist_ok=True)
    json_path = rf'dataset\{class_name}\{class_name}\segment_0.json'


    web_cam(save_folder=save_folder)
    node_json(input_video, save_folder ,show=True)
    split_json(input_json, output_folder, time_mark_path)

    if visualized:
        visualize(json_path)

    vector = []
    for _ in range(sample_num):
        x = load_json(json_path, mean, std, augment=True).to('cuda')
        prediction = model(x)
        vector.append(prediction.squeeze().tolist())

    # return calculate_cosine_similarity(vector)

def sort(array1, array2):
    pair = []
    paired_arrays = list(zip(array1, array2))
    sorted_arrays = sorted(paired_arrays, key=lambda x: x[0], reverse=True)
    sorted_array1, sorted_array2 = zip(*sorted_arrays)

    for item in zip(sorted_array1, sorted_array2):
        pair.append(item)
    return pair

# def show_boxplot(x, labels, title):
#     plt.figure(figsize=(16, 6))
#     plt.boxplot(x, labels=labels)
#     plt.title(title)
#     plt.show()

def calculate_similarity(vector, class_features, similarity_calculater):
    id_vector = [value for value in class_features.items()]
    similarity_list, id_list = [], []
    for id, vector_i in id_vector:
        # print(id, len(vector_i))
        container = []
        for feature_i in vector_i:
            container.append(similarity_calculater(vector1=np.array(vector), vector2=np.array(feature_i)))

        similarity_list.append(container)
        id_list.append(id)
    return similarity_list, id_list

def sort_list(similarity_list, id_list):
    max_list = []
    for container in similarity_list:
        max_list.append(max(container))
        
    return sort(max_list, id_list)

def get_class_features(dataset_dir, output, mean, std, model, sample_num):#revise
    class_features = {}
    for index in tqdm(range(len(list_folders(dataset_dir)))):
        class_name_list =  list_folders(dataset_dir)
        class_name = class_name_list[index]

        json_path = rf'{dataset_dir}\{class_name}\{class_name}\segment_{0}.json'
        # visualize(json_path)
        x = load_json(json_path)
        try:
            prediction_list = []
            for _ in range(sample_num):
                prediction = video_process_stream_class(model=model,
                                    x=x,
                                    mean=mean,
                                    std=std,
                                    augment=True)
                prediction_list.append(prediction.squeeze().tolist())

            class_features[f'{class_name}'] = prediction_list
        except:
            pass
    with open(output, 'w') as f:
        json.dump(class_features, f, indent=4, ensure_ascii=False)

def get_experiment_features(dataset_dir, output, mean, std, model, sample_num = 1):#revise
    class_features = {}
    for index in tqdm(range(len(list_folders(dataset_dir)))):
        prediction_list = []
        for i in range(sample_num):
            prediction, class_name = video_process_stream_expriment(model=model,
                                dataset_dir=dataset_dir,
                                index=index,
                                mean=mean,
                                std=std,
                                augment=False,
                                i=i)
            prediction_list.append(prediction.squeeze().tolist())

        class_features[f'{class_name}'] = prediction_list
        with open(output, 'w') as f:
            json.dump(class_features, f, indent=4, ensure_ascii=False)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from utils import cosine_similarity, calculate_cosine_similarity, sort_list
import json
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 修改中文字體
plt.rcParams['axes.unicode_minus'] = False # 顯示負號

def integrate_commomword(mean, std, model, name, catagory, calculater):
    get_class_features(dataset_dir = r'data-alpha\commom word2',
                       output = r'features\feature.json',
                       mean=mean,
                       std=std,
                       model=model,
                       sample_num=25)
    get_experiment_features(dataset_dir = 'data-alpha\expriment-commom word2',
                            output = 'features\expriment.json',
                            mean=mean,
                            std=std,
                            model=model)

    with open('delete_list.json', 'r') as f:
        delete_List = json.load(f)

    with open(r'features\expriment.json', 'r') as f:
        expriment_features = json.load(f)
    for tag in delete_List:
        try:
            del expriment_features[tag]
        except:
            pass

    id_vector = [value for value in expriment_features.items()]

    with open(r'features\feature.json', 'r') as f:
        class_features = json.load(f)
    for tag in delete_List:
        try:
            del class_features[tag]
        except:
            pass
    matrix = []
    for id, vector_list in tqdm(id_vector):
        for i, vector_i in enumerate(vector_list):

            similarity_list, id_list = calculate_similarity(vector_i, class_features, calculater)
            x = sort_list(similarity_list, id_list)
            matrix.append({'index':i,'tag':id,'list':x})
    y_pred = []
    y_true = []
    for item in matrix:
        y_pred.append(item['list'][0][1])
        y_true.append(item['tag'])
    conf_mat = confusion_matrix(y_true, y_pred, labels=id_list)
    plt.figure(figsize=(12, 10))
    # 使用Seaborn繪製混淆矩陣
    sns.heatmap(conf_mat, annot=False, cmap="Blues")

    # 隱藏軸標籤
    plt.xticks([])
    plt.yticks([])

# 添加標籤
    plt.ylabel('true lables')
    plt.xlabel('prediction lables')

    plt.savefig(rf'log\{catagory}\{name}.png')
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')  # 'macro' 表示未加權平均
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    with open(f'log/{catagory}_indicator.json','r') as f:
        matrix = json.load(f)
    matrix[name] = {'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1}
    with open(f'log/{catagory}_indicator.json','w') as f:
        json.dump(matrix,f,ensure_ascii=False)

def integrate_notation(mean, std, model, name, catagory, calculater):
    get_class_features(dataset_dir = r'data\phonetic notation',
                       output = r'features\feature-phonetic notation2.json',
                       mean=mean,
                       std=std,
                       model=model,
                       sample_num=100)
    get_experiment_features(dataset_dir = 'data\expriment-phonetic notation',
                            output = 'features\expriment-phonetic notation2.json',
                            mean=mean,
                            std=std,
                            model=model,
                            sample_num = 10)

    with open('delete_list.json', 'r') as f:
        delete_List = json.load(f)

    with open(r'features\expriment-phonetic notation2.json', 'r') as f:
        expriment_features = json.load(f)
    for tag in delete_List:
        try:
            del expriment_features[tag]
        except:
            pass

    id_vector = [value for value in expriment_features.items()]

    with open(r'features\feature-phonetic notation2.json', 'r') as f:
        class_features = json.load(f)
    for tag in delete_List:
        try:
            del class_features[tag]
        except:
            pass
    matrix = []
    for id, vector_list in tqdm(id_vector):
        for i, vector_i in enumerate(vector_list):

            similarity_list, id_list = calculate_similarity(vector_i, class_features, calculater)
            x = sort_list(similarity_list, id_list)
            matrix.append({'index':i,'tag':id,'list':x})
    y_pred = []
    y_true = []
    for item in matrix:
        y_pred.append(item['list'][0][1])
        y_true.append(item['tag'])
    conf_mat = confusion_matrix(y_true, y_pred, labels=id_list)
    plt.figure(figsize=(12, 10))
    # 使用Seaborn繪製混淆矩陣
    sns.heatmap(conf_mat, annot=True, cmap="Blues", xticklabels= id_list, yticklabels= id_list)


# 添加標籤
    plt.ylabel('true lables')
    plt.xlabel('prediction lables')

    plt.savefig(rf'log\{catagory}\{name}.png')
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')  # 'macro' 表示未加權平均
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    with open(f'log/{catagory}_indicator.json','r') as f:
        matrix = json.load(f)
    matrix[name] = {'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1}
    with open(f'log/{catagory}_indicator.json','w') as f:
        json.dump(matrix,f,ensure_ascii=False)