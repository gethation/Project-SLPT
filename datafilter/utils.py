import numpy as np
import json
import cv2

def coordinate_transform(coordinate):
    coordinate = np.array(coordinate)
    container = np.empty((64, 40, 2))
    container[:, :20, :] = coordinate[:, :20, :]
    container[:, 20:, :] = coordinate[:, 21:41, :]
    return container

def elongate(keypoint_coordinates_dim, new_length):

    new_indices = np.linspace(0, len(keypoint_coordinates_dim) - 1, new_length)
    new_array = np.interp(new_indices, np.arange(len(keypoint_coordinates_dim)), keypoint_coordinates_dim)

    return new_array

def extend(keypoint_coordinates, new_length):
    filled_keypoint_coordinates = np.zeros((new_length, keypoint_coordinates.shape[1], 2))
    for i in range(keypoint_coordinates.shape[1]):
        filled_keypoint_coordinates[:, i, 0] = elongate(keypoint_coordinates[:, i, 0], new_length)
        filled_keypoint_coordinates[:, i, 1] = elongate(keypoint_coordinates[:, i, 1], new_length)

    return filled_keypoint_coordinates

x = [(i, i+1) for i in range(0,4)]+[(i, i+1) for i in range(5,8)]+[(i, i+1) for i in range(9,12)]+[(i, i+1) for i in range(13,16)]+[(i, i+1) for i in range(17,19)]+[(0,5),(0,17),(5,9),(9,13),(13,17)]
connections = [i for i in x]+[ (i[0]+21, i[1]+21) for i in x] + [(i[0]+42, i[1]+42) for i in [(12,11),(12,14),(11,13),(13,0-42),(14,21-42)]]

# x = [(i, i+1) for i in range(0,4)]+[(i, i+1) for i in range(5,8)]+[(i, i+1) for i in range(9,12)]+[(i, i+1) for i in range(13,16)]+[(i, i+1) for i in range(17,19)]+[(0,5),(0,17),(5,9),(9,13),(13,17)]
# connections = [i for i in x]+[ (i[0]+20, i[1]+20) for i in x] + [(i[0]+40, i[1]+40) for i in [(1,2),(2,4),(1,3),(3,0-40),(4,20-40)]]
# connections = []


def visualize(coordinates, lenth=64):
    # coordinates = takeout_zero(coordinates)
    # coordinates = random_rotate(coordinates)
    # coordinates = random_scaling(coordinates)
    # coordinates = offsetalize(coordinates)
    keypoint_coordinates = extend(coordinates, lenth).astype(np.int16)
    print(coordinates.shape, keypoint_coordinates.shape)
    # 创建黑色背景
    background = 255 * np.ones((640, 640, 3), dtype=np.uint8)

    # 指定要连接的点的索引

    for keypoints in keypoint_coordinates:
        # 创建黑色背景的副本
        frame = background.copy()

        # 绘制关键点
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint[0], keypoint[1]
            if i <= 20:
                color = (255, 0, 0)
            elif i > 20 and i <= 40:
                color = (0, 0, 255)
            else: color = (0, 0, 0)
                
            if i == 41 or i == 20:
                pass
            else:
                cv2.circle(frame, (x, y), 5, color, -1)

        # 连接指定的点
        for connection in connections:
            try:
                start_point = tuple(keypoints[connection[0]])
                end_point = tuple(keypoints[connection[1]])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            except:
                pass

        cv2.imshow('Hand Key Points', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
