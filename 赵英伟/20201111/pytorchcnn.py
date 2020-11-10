import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 5
LR = 0.001          # 学习率

x=np.load("C:/Users/Administrator/Desktop/data/data.npy")
y=np.load("C:/Users/Administrator/Desktop/data/label.npy")
# x=np.load("C:/Users/Administrator/Desktop/data/dataset_inputs.npy")
# y=np.load("C:/Users/Administrator/Desktop/data/dataset_outputs.npy")
# print(x)
# print(y)

X = x.astype(np.float32)
y =y[:,0]
y = y.astype(np.long)

torch_x_data = torch.from_numpy(X)#np转为torch可用的数据\
torch_y_data = torch.from_numpy(y)

print(torch_x_data.shape)
print(torch_y_data.shape)
torch_dataset = Data.TensorDataset(torch_x_data, torch_y_data)


loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 打乱数据 (打乱比较好)
)

#---------------------模型两层CNN加全连接层分类两类------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 19, 52)
            nn.Conv1d(
                in_channels=52,      # input height
                out_channels=16,    # n_filters
                kernel_size=1,      # filter size
                stride=1,           # filter movement/step

            ),
            nn.ReLU(),    # activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 1, 1),
            nn.ReLU(),  # activation
        )
        self.out = nn.Linear(19*32,2)   # fully connected layer, output 2 classes

    def forward(self, x):

        x = self.conv1(x.permute(0,2,1))#维度转换了位置
        x = self.conv2(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图
        #print(x.shape)
        output = self.out(x.permute(0, 1))
        #print(output.shape)
        return output

cnn = CNN()
print(cnn)  # net architecture
#-------------------------------训练-------------------------------------------
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):   # 分配 batch data, normalize x when iterate train_loader
        output = cnn(b_x)# cnn output这里是最后的预测
        loss = loss_func(output, b_y.long())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients"""
        print('Epoch: ', epoch, '| Step: ', step, "|loss:",loss)
