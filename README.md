# CNN-handwrite
## 基于Python的CNN卷积神经网络构成的简易手写数字识别<br/>Simple handwritten digit recognition based on Python CNN simple neural network

### 所需要第三方库 Third-party libraries needed to run this programm

- tkinter

- Pillow

- torch

- torchvision
### 数据增强 Data Augmentation

(line:102-109)

- Resize

- RandomRotation

- RandomAffine

- RandomHorizontalFlip

- ColorJitter

- ToTensor

- Normalize

- RandomErasing

## 1.首先, 安装第三方库 Install Third-party libraries

```
pip install Pillow
pip install torch torchvision
```

## 2.配置好代码 Configure the code(CNNcode.py)

batch_size(line:117)
```
return DataLoader(combined_dataset, batch_size=5000, shuffle=is_train, drop_last=True)
```

optimizer(line:146)
```
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-4)
```

epoch(line:150)
```
for epoch in range(100):
```

Stop early(line:164)
```
if accuracy > 0.97:
```

## 3.运行 Run this programm


