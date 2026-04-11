import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

myimg = Image.open("first_assignments/Lenna.png")
mygray = np.array(myimg.convert("L"), dtype=np.float32)
plt.figure(figsize = (6,6))
plt.imshow(mygray, cmap = plt.cm.gray)
plt.axis("off")
plt.show()


imh , imw = mygray.shape
mygray = torch.from_numpy(mygray.reshape(1,1,imh,imw))
print(mygray.shape)

kersize = 5 
ker= torch.ones(kersize,kersize , dtype = torch.float32)*-1
ker[2,2] = 24
ker = ker.reshape((1,1,kersize,kersize))
conv2d = nn.Conv2d(1,2,(kersize,kersize),bias = False)

conv2d.weight.data[0] = ker


imconv2dout = conv2d(mygray)
imconv2dout_im = imconv2dout.data.squeeze()
print(imconv2dout_im.shape)
print(conv2d.weight.data)
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.imshow(imconv2dout_im[0], cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(imconv2dout_im[1], cmap = plt.cm.gray)
plt.axis("off")
plt.show()



maxpool2 = nn.MaxPool2d(2 , stride = 2)
pool2_out = maxpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze().detach()
print(pool2_out_im.shape)
torch.Size([1,2,254,254])
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.imshow(pool2_out_im[0], cmap = plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(pool2_out_im[1], cmap = plt.cm.gray)
plt.axis("off")
plt.show()


avgpool2 = nn.AdaptiveAvgPool2d(output_size=(100,100))
pool2_out = avgpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze().detach()
print(pool2_out_im.shape)

