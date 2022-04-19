import torch
import torch.nn as nn
import hiddenlayer as hl
import torch.optim as optim
from torch.autograd import Variable
from dataset import read_image
from dataset import create_dataset
from networks import DeTurbAutoEncoder

# 数据读取
data_path_train = r"G:\dataset\MyDataset\train-Indexed"
data_path_val = r"G:\dataset\MyDataset\val-Indexed"
dataset_train = read_image(data_path_train, 18)
dataset_val = read_image(data_path_val, 18)
data_loader_train = create_dataset(dataset_train, 4, True)
data_loader_val = create_dataset(dataset_val, 4, True)
# 调用gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 模型初始化
DAEmodel = DeTurbAutoEncoder().to(device)
# 定义优化器
LR = 0.0003
optimizer = torch.optim.Adam(DAEmodel.parameters(), lr=LR)
# 定义损失函数
loss_func = nn.MSELoss()
# 记录训练过程的指标
history1 = hl.History()
# 使用canvas进行可视化
canvas1 = hl.Canvas()
train_num = 0
val_num = 0
# 对模型进行迭代训练，对所有数据训练epoch轮
for epoch in range(10):
    train_loss_epoch = 0
    val_loss_epoch = 0
    # 对训练数据的加载器进行迭代计算
    for step, data in enumerate(data_loader_train):
        DAEmodel.train()
        input = Variable(data['input']).cuda()
        truth = Variable(data['truth']).cuda()
        # 使用每个batch进行模型训练
        _, output = DAEmodel(input)
        loss = loss_func(output, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * input.size(0)
        train_num += input.size(0)
    # 使用每个batch进行验证模型
    for step, data in enumerate(data_loader_val):
        DAEmodel.eval()
        input = Variable(data['input']).cuda()
        truth = Variable(data['truth']).cuda()
        _, output = DAEmodel(input)
        loss = loss_func(output, truth)
        val_loss_epoch += loss.item() * input.size(0)
        val_num += input.size(0)
    # 计算一个epoch的损失
    train_loss = train_loss_epoch / train_num
    val_loss = val_loss_epoch / val_num
    # 保存每个epoch上的输出loss
    history1.log(epoch, train_loss=train_loss, val_loss=val_loss)
    with canvas1:
        canvas1.draw_plot([history1["train_loss"], history1["val_loss"]])
