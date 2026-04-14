import torch 
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as data 

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


boston_x , boston_y = load_boston(return_X_y=True)
print(boston_x.shape)
plt.figure()
plt.hist(boston_y, bins=20)
plt.show()

ss = StandardScaler(with_mean=True, with_std=True)
boston_xs = ss.fit_transform(boston_x)

train_xt = torch.from_numpy(boston_xs.astype(np.float32))

train_yt = torch.from_numpy(boston_y.astype(np.float32))

train_data = data.TensorDataset(train_xt , train_yt)

train_loader = data.DataLoader(dataset=train_data, batch_size=128, shuffle=True ,num_workers=1)

class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(in_features=13  , out_features=10 , bias = True) , nn.ReLU() , nn.Linear(10,10) , nn.ReLU())
        self.regression = nn.Linear(10,1)

    
    def forward(self, x):
     x= self.hidden(x)
     output = self.regression(x)

     return output 
    

mlp2 = MLPmodel2()
print( mlp2)
optimizer = SGD(mlp2.parameters(), lr=0.001)
loss_func  = nn.MSELoss()
train_loss_all = []

for epoch in range(30):
   for step , (batch_x , batch_y) in enumerate(train_loader):
         output = mlp2(batch_x).flatten()
         train_loss = loss_func(output , batch_y)
         optimizer.zero_grad()
         train_loss.backward()
         optimizer.step()   
         train_loss_all.append(train_loss.item())

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(train_loss_all , "r-")
plt.title("训练损失曲线")
plt.show()
