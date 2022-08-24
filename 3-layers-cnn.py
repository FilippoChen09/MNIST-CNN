import torch
import torchvision

# 设置超参数
num_epochs = 5
batch_size = 100
num_classes = 10
learning_rate = 0.001

# 从TorchVision下载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor())

# 使用PyTorch提供的DataLoader，以分批乱序加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# 构建卷积神经网络
class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, in_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)

        # 将卷积层的结果拉成向量再通过全连接层
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# 实例化模型（可调用对象）
model = ConvolutionalNeuralNetwork(num_classes)

# 设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 检验模型在测试集上的准确性
correct = 0
total = 0
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy on test set: {} %'.format(100 * correct / total))

#torch.save(model.state_dict(), './model.ckpt')
