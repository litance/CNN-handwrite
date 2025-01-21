import tkinter as tk
from tkinter import Canvas
from PIL import ImageGrab, ImageOps, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import MNIST, EMNIST, KMNIST
import os

#Define GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Define CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3

        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 3 * 3)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


#Create handwritting panel GUI
def create_handwriting_panel(net):
    def predict_from_canvas():
        try:
            x = root.winfo_rootx() + canvas.winfo_x()
            y = root.winfo_rooty() + canvas.winfo_y()
            x1 = x + canvas.winfo_width()
            y1 = y + canvas.winfo_height()
            image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
            image = ImageOps.invert(image)
            image = image.resize((28, 28))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)
            net.eval()
            with torch.no_grad():
                prediction = torch.argmax(net(image_tensor))
            result_label.config(text=f"Prediction: {int(prediction)}")
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

    def clear_canvas():
        canvas.delete("all")
        draw_grid()

    def paint(event):
        x1, y1 = event.x // 20 * 20, event.y // 20 * 20
        x2, y2 = x1 + 20, y1 + 20
        canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

    def draw_grid():
        for i in range(25):
            canvas.create_line(i * 20, 0, i * 20, 500, fill="gray")
            canvas.create_line(0, i * 20, 500, i * 20, fill="gray")

    root = tk.Tk()
    root.title("Handwriting Recognition")
    canvas = Canvas(root, width=500, height=500, bg="white")
    canvas.grid(row=0, column=0, columnspan=2)
    draw_grid()
    canvas.bind("<B1-Motion>", paint)
    predict_button = tk.Button(root, text="Predict", command=predict_from_canvas)
    predict_button.grid(row=1, column=0)
    clear_button = tk.Button(root, text="Clear", command=clear_canvas)
    clear_button.grid(row=1, column=1)
    result_label = tk.Label(root, text="Prediction: ")
    result_label.grid(row=2, column=0, columnspan=2)
    root.mainloop()


#Get data loader
def get_data_loader(is_train):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.5)
    ])
    datasets = [
        MNIST(root="./data", train=is_train, transform=transform, download=True),
        EMNIST(root="./data", split="balanced", train=is_train, transform=transform, download=True),
        KMNIST(root="./data", train=is_train, transform=transform, download=True)
    ]
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=5000, shuffle=is_train, drop_last=True)


#Evaluate
def evaluate(test_data, net):
    net.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            # For EMNIST and KMNIST, ensure label range is within 0-9
            if y.max() >= 10:  # Handle case for multi-class datasets like EMNIST and KMNIST
                y = y % 10  # Make labels fall in 0-9 range
            outputs = net(x)
            n_correct += (torch.argmax(outputs, dim=1) == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total


#Main
def main():
    net = CNN().to(device)
    if os.path.exists("cnn.pth"):
        net.load_state_dict(torch.load("cnn.pth"))
        print("Model loaded from cnn.pth")
    else:
        train_data = get_data_loader(is_train=True)
        test_data = get_data_loader(is_train=False)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

        net.train()
        for epoch in range(100):
            for (x, y) in train_data:
                x, y = x.to(device), y.to(device)
                # For EMNIST and KMNIST, ensure label range is within 0-9
                if y.max() >= 10:  # Handle case for multi-class datasets like EMNIST and KMNIST
                    y = y % 10  # Make labels fall in 0-9 range
                optimizer.zero_grad()
                output = net(x)
                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()
            accuracy = evaluate(test_data, net)
            print(f"Epoch {epoch}, accuracy: {accuracy:.4f}")
            scheduler.step(accuracy)
            if accuracy > 0.97:
                print("Stopping early: High accuracy achieved.")
                break
        torch.save(net.state_dict(), "cnn.pth")
        print("Model saved as cnn.pth")

    create_handwriting_panel(net)

#Start training
if __name__ == "__main__":
    main()
