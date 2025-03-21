import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.resnet50 import ResNet50  # 导入标准ResNet50模型
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet  # 确保你使用正确的测试集

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建ResNet50模型
model = ResNet50().to(device)

# 加载训练好的模型
checkpoint = torch.load('../model_data/resmodel-50-0.8198.pt')  # 训练好的模型路径

# 加载模型的状态字典（忽略fc层）
model.load_state_dict(checkpoint['model'], strict=False)

# 重新初始化fc层，以匹配当前的类别数（假设是10类）
model.fc = torch.nn.Linear(2048, 10)  # 重新定义fc层，输出10类

# 加载新的fc层权重（如果有的话），否则直接重新初始化
model.fc.weight.data.normal_(0, 0.01)  # 对新fc层的权重进行初始化
model.fc.bias.data.zero_()  # 对新fc层的偏置进行初始化

model.eval()  # 设置模型为评估模式

# 数据加载器
dataset_test = ChineseFoodNetTestSet()  # 使用你自己的数据集路径
dataloader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)

# 准备存储准确率
correct_top1 = 0
correct_top5 = 0
total = 0

# 计算Top-1和Top-5准确率
with torch.no_grad():
    for data in dataloader_test:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # 模型推理
        outputs = model(inputs)

        # 计算Top-1准确率
        _, predicted_top1 = torch.max(outputs, 1)

        # 计算Top-5准确率
        _, predicted_top5 = outputs.topk(5, dim=1)

        total += targets.size(0)
        correct_top1 += (predicted_top1 == targets).sum().item()

        # Top-5 Accuracy: Check if target is in top 5 predictions
        correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

# 输出Top-1和Top-5准确率
top1_acc = 100 * correct_top1 / total
top5_acc = 100 * correct_top5 / total

print(f'Top-1 Accuracy: {top1_acc:.2f}%')
print(f'Top-5 Accuracy: {top5_acc:.2f}%')