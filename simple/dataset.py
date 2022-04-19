import os
import cv2
import torch
from torch.utils.data import DataLoader


def read_image(data_path, channel_num=3):
    img_input = []
    img_truth = []
    for filename in os.listdir(os.path.join(data_path, 'input')):
        img_input_path = os.path.join(data_path, 'input', filename)
        img_truth_path = os.path.join(data_path, 'truth', filename)
        if channel_num == 3:
            Input_temp = cv2.imread(img_input_path)
            Truth_temp = cv2.imread(img_truth_path)
        else:
            Input_temp = cv2.imread(img_input_path, 0)
            Truth_temp = cv2.imread(img_truth_path, 0)
        # print('img:', img.shape)
        Input_temp = torch.from_numpy(Input_temp)
        Truth_temp = torch.from_numpy(Truth_temp)
        Input_temp = torch.unsqueeze(Input_temp, dim=0)
        Truth_temp = torch.unsqueeze(Truth_temp, dim=0)
        # print('img:', img.shape)
        img_input.append(Input_temp)
        img_input.append(Truth_temp)
    img_input = torch.cat(img_input, dim=0)
    img_truth = torch.cat(img_truth, dim=0)
    Height = img_input.shape[-2]
    Width = img_input.shape[-1]
    input = torch.reshape(img_input, (-1, channel_num, Height, Width))
    truth = torch.reshape(img_truth, (-1, channel_num, Height, Width))
    return {'input': input, 'truth': truth}


# images = read_image(r"G:\dataset\MyDataset\test\turb4", 18)
# print(images.shape)


def create_dataset(dataset, batchsize, shuffle):
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    return dataloader
