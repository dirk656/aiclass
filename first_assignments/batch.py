import torch 
import torch.utils.data as data
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

train_data_tramsform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data_dir = "first_assignments/imagedata"
train_data = ImageFolder(train_data_dir, transform = train_data_tramsform)
train_data_loader = data.DataLoader(train_data, batch_size = 16, shuffle = True,num_workers=1)
print(train_data.targets)

for step , (b_x , b_y) in enumerate(train_data_loader):
    if step > 0:
        break

print(b_x.shape)
print(b_y.shape)
print(b_x.min(), b_x.max())

