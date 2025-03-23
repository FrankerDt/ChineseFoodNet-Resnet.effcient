import argparse
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from utils.ChineseFoodNetSet import ChineseFoodNetTestSet, ChineseFoodNetValSet
from model.efficient_cbam import EfficientCBAMResNet
import datetime

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    args = parser.parse_args()

    # 初始化设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取所有模型文件路径
    model_folder = 'model_data_test'
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pt')]

    # 保存所有模型的精度结果
    results = []

    # ==================== 计算测试集精度 ============================
    dataset_test = ChineseFoodNetTestSet()
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.threads)

    # ==================== 遍历模型进行测试 ============================
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        print(f"Testing model: {model_file}")

        # 加载efficient_cbam_resnet模型
        model = EfficientCBAMResNet(num_classes=208).to(device)

        # 加载模型权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.eval()

        # 计算测试集精度 (Top 1 & Top 5)
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader_test:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                _, predicted_top5 = outputs.topk(5, dim=1)

                total += targets.size(0)
                correct_top1 += (predicted == targets).sum().item()
                correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

        top1_acc_test = 100 * correct_top1 / total
        top5_acc_test = 100 * correct_top5 / total

        # 计算验证集精度 (Val Top 1 & Val Top 5)
        dataset_val = ChineseFoodNetValSet()
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.threads)

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                _, predicted_top5 = outputs.topk(5, dim=1)

                total += targets.size(0)
                correct_top1 += (predicted == targets).sum().item()
                correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()

        top1_acc_val = 100 * correct_top1 / total
        top5_acc_val = 100 * correct_top5 / total

        # 打印当前模型的精度结果
        print(f"Model: {model_file}")
        print(f"Test Top 1 Accuracy: {top1_acc_test:.2f}%, Test Top 5 Accuracy: {top5_acc_test:.2f}%")
        print(f"Val Top 1 Accuracy: {top1_acc_val:.2f}%, Val Top 5 Accuracy: {top5_acc_val:.2f}%")
        print("-" * 50)

        # 保存每个模型的精度结果
        results.append((model_file, top1_acc_test, top5_acc_test, top1_acc_val, top5_acc_val))

    # 根据精度排序并选择最优模型
    top1_test_sorted = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    top5_test_sorted = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    val_sorted = sorted(results, key=lambda x: x[3], reverse=True)[:3]
    val_top5_sorted = sorted(results, key=lambda x: x[4], reverse=True)[:3]

    # 打印最优模型
    print("\nTop 3 models based on Test Top 1:")
    for idx, (model_name, top1_test, top5_test, top1_val, top5_val) in enumerate(top1_test_sorted):
        print(
            f"{idx + 1}. Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%")

    print("\nTop 3 models based on Test Top 5:")
    for idx, (model_name, top1_test, top5_test, top1_val, top5_val) in enumerate(top5_test_sorted):
        print(
            f"{idx + 1}. Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%")

    print("\nTop 3 models based on Val Top 1:")
    for idx, (model_name, top1_test, top5_test, top1_val, top5_val) in enumerate(val_sorted):
        print(
            f"{idx + 1}. Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%")

    print("\nTop 3 models based on Val Top 5:")
    for idx, (model_name, top1_test, top5_test, top1_val, top5_val) in enumerate(val_top5_sorted):
        print(
            f"{idx + 1}. Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%")

    # ==================== 保存结果到文件 ============================
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    result_folder = 'model_test_res'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_file = os.path.join(result_folder, f"test_results_{current_date}.txt")
    with open(result_file, 'w') as f:
        f.write("Top 3 models based on Test Top 1:\n")
        for model_name, top1_test, top5_test, top1_val, top5_val in top1_test_sorted:
            f.write(
                f"Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%\n")

        f.write("\nTop 3 models based on Test Top 5:\n")
        for model_name, top1_test, top5_test, top1_val, top5_val in top5_test_sorted:
            f.write(
                f"Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%\n")

        f.write("\nTop 3 models based on Val Top 1:\n")
        for model_name, top1_test, top5_test, top1_val, top5_val in val_sorted:
            f.write(
                f"Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%\n")

        f.write("\nTop 3 models based on Val Top 5:\n")
        for model_name, top1_test, top5_test, top1_val, top5_val in val_top5_sorted:
            f.write(
                f"Model: {model_name}, Test Top 1: {top1_test:.2f}%, Test Top 5: {top5_test:.2f}%, Val Top 1: {top1_val:.2f}%, Val Top 5: {top5_val:.2f}%\n")

    print(f"Results saved to {result_file}")
