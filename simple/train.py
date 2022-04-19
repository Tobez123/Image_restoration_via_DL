import torch
import torch.nn as nn
import hiddenlayer as hl
import torch.optim as optim
from dataset import read_image
from dataset import create_dataset
from networks import DeTurbAutoEncoder

# 数据读取
data_path = r"G:\dataset\MyDataset\train-Indexed"
dataset = read_image(data_path, 18)
data_loader = create_dataset(dataset, 4, True)
# 模型初始化
DAEmodel = DeTurbAutoEncoder()
# 定义优化器
LR = 0.0003
optimizer = torch.optim.Adam(DAEmodel.parameters(), lr=LR)
# 定义损失函数
loss_func = nn.MSELoss()
history1 = hl.History()
canvas1 = hl.Canvas()
train_num = 0
val_num = 0
# 对模型进行迭代训练，对所有数据训练epoch轮
for epoch in range(10):
    train_loss_epoch = 0
    val_loss_epoch = 0
