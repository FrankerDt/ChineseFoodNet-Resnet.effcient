import torch
from torch.utils.data import DataLoader
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet
from model.efficient_cbam import EfficientCBAMResNet
import argparse

def top_k_accuracy(output, target, k=5):
    """计算top-k准确率"""
    with torch.no_grad():
        max_k = max((1, k))
        batch_size = target.size(0)
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        return correct_k / batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model_data/efficient_cbam/efficient_cbam-62-0.7497.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=208)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = EfficientCBAMResNet(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("Model loaded successfully.")

    # 加载测试集
    test_dataset = ChineseFoodNetTestSet()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.topk(5, 1, True, True)

            total += targets.size(0)
            correct_top1 += (predicted[:, 0] == targets).sum().item()
            correct_top5 += (predicted == targets.unsqueeze(1)).sum().item()

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
