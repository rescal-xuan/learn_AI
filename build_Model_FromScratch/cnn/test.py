import PIL.Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from  torch.utils.data import Dataset
from  torchvision  import datasets
import os
img_path = 'dataset/dogvscat/test/dogs/dog.4001.jpg'
classes=['cats', 'dogs']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and its state
model_dir = 'pretrained_models/efficientnet-b8.pth'



# 下载并加载预训练模型
model = EfficientNet.from_name('efficientnet-b8')
model.load_state_dict(torch.load(model_dir))
model._fc.out_features =3
model = model.to(device)

checkpoint = torch.load('./model_pt/model.pt', map_location=torch.device('cpu'))
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)  # directly loading a state dict

model.eval()

# Load and preprocess the image
try:
    img = PIL.Image.open(img_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: Image file not found at {img_path}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.52, 0.52, 0.52))
])

img = transform(img).unsqueeze(0).to(device)

# Make the prediction
with torch.no_grad():
    out = model(img)

    pred = torch.argmax(out, -1)  
    print(classes[pred])