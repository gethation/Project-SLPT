import torch
import torch.nn as nn
from lightly.models import utils
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
import cv2
import numpy as np
import datetime
import wandb
import zipfile
import shutil
from tqdm.auto import tqdm
# 逆归一化操作

x = [(i, i+1) for i in range(0,4)]+[(i, i+1) for i in range(5,8)]+[(i, i+1) for i in range(9,12)]+[(i, i+1) for i in range(13,16)]+[(i, i+1) for i in range(17,19)]+[(0,5),(0,17),(5,9),(9,13),(13,17)]
connections = [i for i in x]+[ (i[0]+20, i[1]+20) for i in x]


def denormalize_data(normalized_tensor, mean, std):
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0], std=[1/std]),
        transforms.Normalize(mean=[-mean], std=[1])
    ])
    denormalized_tensor = denormalize(normalized_tensor)
    return denormalized_tensor

def video_show(visual_prediction, visual_target, mean, std, output_file):
    
    prediction_coordinate = denormalize_data(visual_prediction, mean, std).cpu()
    # prediction_coordinate = visual_prediction
    prediction_coordinate = prediction_coordinate.detach().numpy()
    prediction_target = denormalize_data(visual_target, mean, std).cpu()
    # prediction_target = visual_target
    prediction_target = prediction_target.detach().numpy()

    # 创建白色背景，大小为 1280x640
    background = 255 * np.ones((640, 1280, 3), dtype=np.uint8)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (1280, 640))
    video = []

    for keypoints_set_1, keypoints_set_2 in zip(prediction_coordinate, prediction_target):
        # 创建黑色背景的副本
        frame = background.copy()

        # 绘制第一组关键点（左边）
        for i, keypoint in enumerate(keypoints_set_1):
            x, y = keypoint[0], keypoint[1]
            if i >= 21:
                color = (0, 0, 255)  # 红色
            else:
                color = (255, 0, 0)  # 蓝色

            x = x.astype(np.int16)
            y = y.astype(np.int16)

            cv2.circle(frame[:, :640], (x, y), 5, color, -1)  # 在左边绘制点

        for connection in connections:
            start_point = tuple(keypoints_set_1[connection[0]])
            end_point = tuple(keypoints_set_1[connection[1]])

            start_point = (int(start_point[0]), int(start_point[1]))
            end_point = (int(end_point[0]), int(end_point[1]))

            cv2.line(frame[:, :640], start_point, end_point, (0, 255, 0), 2)

        # 绘制第二组关键点（右边）
        for i, keypoint in enumerate(keypoints_set_2):
            x, y = keypoint[0], keypoint[1]
            if i >= 21:
                color = (0, 0, 255)  # 绿色
            else:
                color = (255, 0, 0)  # 深蓝色

            x = x.astype(np.int16)
            y = y.astype(np.int16)
            cv2.circle(frame[:, 640:], (x, y), 5, color, -1)  # 在右边绘制点

        for connection in connections:
            start_point = tuple(keypoints_set_2[connection[0]])
            end_point = tuple(keypoints_set_2[connection[1]])

            start_point = (int(start_point[0]), int(start_point[1]))
            end_point = (int(end_point[0]), int(end_point[1]))

            cv2.line(frame[:, 640:], start_point, end_point, (0, 255, 0), 2)

        # 将当前帧写入视频
        out.write(frame)
        video.append(frame)
        # cv2.imshow('Astype Key Points', frame)

        # if cv2.waitKey(100) & 0xFF == ord('q'):
        #     break

    # 释放视频写入器和窗口
    # out.release()
    # cv2.destroyAllWindows()
    return np.array(video).transpose((0, 3, 1, 2)).astype(np.int8)




# 定义归一化转换
def normalize_data(tensor, mean, std):
    # mean = tensor.mean()
    # std = tensor.std()
    normalize = transforms.Normalize(mean=[mean], std=[std])
    normalized_tensor = normalize(tensor)
    return normalized_tensor

def load_mean_std(data_folder):
    # 加载文件夹中的所有 JSON 文件
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    sample_num = int(len(json_files)*0.01)

    json_files = np.random.choice(json_files, size=sample_num, replace=False)
    # 创建一个存储所有数据的列表
    all_data = []

    # 遍历所有 JSON 文件并加载数据
    for json_file in json_files:
        with open(os.path.join(data_folder, json_file), 'r') as file:
            data = json.load(file)
            all_data.append(data)
        
    # 转换为张量
    tensor_data = torch.tensor(all_data, dtype=torch.float32)

    # 归一化数据
    mean, std = tensor_data.mean(), tensor_data.std()

    return mean, std

# 创建Dataset和DataLoader
class Node_Dataset(Dataset):
    def __init__(self, data_folder, mean, std):
        self.mean = mean
        self.std = std
        self.data_folder = data_folder
        self.json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):

        json_file = self.json_files[idx]

        with open(os.path.join(self.data_folder, json_file), 'r') as file:
            x = json.load(file)

        data = augmentation(x)
        data = torch.as_tensor(torch.from_numpy(data), dtype=torch.float32)
        batch = normalize_data(data, self.mean, self.std)
        return batch

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
    angle = np.radians(np.random.randint(-20, 20))
    center = [320, 320]
    rotated_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            rotated_coordinates[i,j] = rotate_point(point, center, angle)
    return rotated_coordinates

def random_scaling(coordinates):
    scale_factor = np.random.uniform(0.7, 1.6)
    center = [320, 320]
    scaled_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            scaled_coordinates[i,j] = scaling_point(point, center, scale_factor)
    return scaled_coordinates

def offsetalize(coordinates):
    offset = np.random.randint(-50, 50, 2)
    offsetalized_coordinates = np.empty_like(coordinates)
    for i, frame in enumerate(coordinates):
        for j, point in enumerate(frame):
            offsetalized_coordinates[i,j] = point + offset
    return offsetalized_coordinates

def augmentation(x):
    coordinates = np.array(x)
    coordinates = random_rotate(coordinates)
    coordinates = random_scaling(coordinates)
    keypoint_coordinates = offsetalize(coordinates)
    return keypoint_coordinates

def get_time():
    current_datetime = datetime.datetime.now()
    current_hour = current_datetime.hour
    current_minute = current_datetime.minute
    second = current_datetime.second

    return f'{current_hour+8}:{current_minute}:{second}'

def log(CHECKPOINT_PATH, model, optimizer, step, mean, std):

    torch.save({ # Save our checkpoint loc
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mean': mean,
        'std': std,
        'step':step
        }, CHECKPOINT_PATH)
    wandb.save(CHECKPOINT_PATH) # saves checkpoint to wandb

class MPJPELoss(nn.Module):
    def __init__(self):
        super(MPJPELoss, self).__init__()

    def forward(self, predicted_joints, ground_truth_joints):

        batch_size = predicted_joints.shape[0]

        predicted_joints = predicted_joints.reshape((batch_size, -1, 80//2, 2))
        ground_truth_joints = ground_truth_joints.reshape((batch_size, -1, 80//2, 2))

        assert predicted_joints.shape == ground_truth_joints.shape, "Input tensors must have the same shape."
        
        # Calculate Euclidean distances along the last dimension (x, y)
        distances = torch.norm(predicted_joints - ground_truth_joints, dim=-1)

        # Calculate mean per joint position error
        mpjpe = torch.mean(distances, dim=2)

        return torch.mean(mpjpe)
def copyfile(file_list, folder=''):
  for file in file_list:
    shutil.copyfile(file, os.path.join(folder,os.path.basename(file)))

def extract_and_split(zip_file_path = '/content/test-dataset.zip',
                      extract_folder = 'dataset',
                      test_folder = 'test_set/test_set',
                      data_folder = 'dataset/test-dataset'):
    
    os.makedirs(test_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    sample_num = int(len(os.listdir(data_folder))*0.001)
    test_json = [os.path.join(data_folder, name) for name in np.random.choice(os.listdir(data_folder), size=sample_num, replace=False)]

    copyfile(test_json, test_folder)

def train_step(model, criterion, batch, optimizer, device='cuda'):
    images = batch.to(device)  # views contains only a single view
    predictions, targets, _, _ = model(images)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.detach()

def Evaluate(model, criterion, testloader, device='cuda'):
    evaluate_loss = 0
    for batch in testloader:
        images = batch.to(device)  # views contains only a single view
        predictions, targets, visual_prediction, visual_target = model(images)
        loss = criterion(predictions, targets)

        evaluate_loss += loss.detach()
    return evaluate_loss / len(testloader), visual_prediction[0], visual_target[0]