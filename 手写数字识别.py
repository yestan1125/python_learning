import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 改进后的模型结构（准确率更高）
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2. 数据加载器
def get_data_loader(is_train):
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root="./data", train=is_train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=is_train)

# 训练循环
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)  # 批次数目，1875（60000/32）
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    for X, y in dataloader:  # 获取图片及其标签
        X, y = X.to(device), y.to(device)
        # 计算预测误差
        pred = model(X)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失
        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新
        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
    train_acc /= size
    train_loss /= num_batches
    return train_acc, train_loss

# 测试准确率评估
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的大小，一共10000张图片
    num_batches = len(dataloader)  # 批次数目，313（10000/32=312.5，向上取整）
    test_loss, test_acc = 0, 0
    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            # 计算loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)
            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()
    test_acc /= size
    test_loss /= num_batches
    return test_acc, test_loss

# 3. 模型训练并保存
def train_and_save_model(model_path):
    train_loader = get_data_loader(True)
    test_loader = get_data_loader(False)
    net = Model().to(device)
    summary(net)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)

    epochs = 4
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []

    print("🚀 开始训练...")
    for epoch in range(epochs):
        net.train()
        epoch_train_acc, epoch_train_loss = train(train_loader, net, loss_fn, optimizer)
        net.eval()
        epoch_test_acc, epoch_test_loss = test(test_loader, net, loss_fn)
        train_acc_list.append(epoch_train_acc)
        train_loss_list.append(epoch_train_loss)
        test_acc_list.append(epoch_test_acc)
        test_loss_list.append(epoch_test_loss)
        print(f'Epoch:{epoch+1:2d}, Train_acc:{epoch_train_acc*100:.1f}%, Train_loss:{epoch_train_loss:.3f}, Test_acc:{epoch_test_acc*100:.1f}%, Test_loss:{epoch_test_loss:.3f}')

    torch.save(net.state_dict(), model_path)
    print(f"✅ 模型已保存到: {model_path}")

# 5. 自定义数据预测并输出 CSV（格式: id,label）
def predict_custom_images(model_path, folder_path, output_csv):
    net = Model().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])

    results = []
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

    for idx, file_name in enumerate(files):
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(tensor)
            pred = torch.argmax(output, dim=1).item()
            results.append((idx, pred))  # index当作id

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(results)

    print(f"📄 CSV 文件已保存：{output_csv}")

# 6. 主执行流程
if __name__ == "__main__":
    model_path = "better_digit_model.pth"
    image_folder = r"C:/Code/Python/Digit_test_dataset-master/test"
    output_csv = "digit_predictions.csv"

    train_and_save_model(model_path)
    predict_custom_images(model_path, image_folder, output_csv)
    