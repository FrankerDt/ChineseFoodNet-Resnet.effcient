import argparse
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet, ChineseFoodNetValSet
from utility.initialize import initialize
from utility.log import Log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Base learning rate at the start of the training.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取model_data_cbam_test文件夹下的所有模型文件路径
    model_folder = 'model_data_cbam_test'
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pt')]

    # ====================计算测试集精度===========================
    dataset_test = ChineseFoodNetTestSet()
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.threads)

    # ====================遍历模型进行测试===========================
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        print(f"Testing model: {model_file}")

        # 加载模型
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for data in dataloader_test:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                _, predicted_top5 = outputs.topk(5, dim=1)

                total += targets.size(0)
                correct_top1 += (predicted == targets).sum().item()
                correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

        top1_acc = 100 * correct_top1 / total
        top5_acc = 100 * correct_top5 / total
        print(f"Model {model_file}: Test Dataset Top 1 Accuracy: {top1_acc:.2f}%, Top 5 Accuracy: {top5_acc:.2f}%")

        # ====================计算验证集精度===========================
        dataset_val = ChineseFoodNetValSet()
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.threads)

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for data in dataloader_val:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                _, predicted_top5 = outputs.topk(5, dim=1)

                total += targets.size(0)
                correct_top1 += (predicted == targets).sum().item()
                correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

        top1_acc = 100 * correct_top1 / total
        top5_acc = 100 * correct_top5 / total
        print(f"Model {model_file}: Val Dataset Top 1 Accuracy: {top1_acc:.2f}%, Top 5 Accuracy: {top5_acc:.2f}%")
