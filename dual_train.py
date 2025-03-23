import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch import nn, optim
from tqdm import tqdm
import os
import wandb
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_encoder import DualEncoderModel
from ChineseFoodPairDataset import ChineseFoodPairDataset

# 对比损失（Contrastive Loss）实现
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        # 计算欧几里得距离
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                          (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# 使用 'if __name__ == "__main__":' 来解决 Windows 上的多线程问题
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    BATCH_SIZE = 32
    EPOCHS = 10
    EMBED_DIM = 256
    LR = 1e-4
    SAVE_PATH = "./dual_model"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # WandB 配置
    wandb.init(project="ChineseFood_classification", name="Dual Encoder Training", config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR
    })

    # 加载 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 加载图文配对数据集
    train_dataset = ChineseFoodPairDataset(
        image_root="./ChineseFoodNet/release_data/train/",
        list_txt_path="./ChineseFoodNet/release_data/train_list.txt",
        class_name_csv="./class_names.csv",
        recipe_json_path="./recipes_chinesefoodnet_207.json"
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 初始化模型，加载你训练好的图像塔权重
    model = DualEncoderModel(embed_dim=256,
                             image_weight_path="model_data/efficient_cbam/efficient_cbam-100-0.7376.pt").to(device)

    # 损失函数：对比损失
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def compute_similarity(image_emb, text_emb):
        """计算图文之间的余弦相似度矩阵"""
        return torch.matmul(image_emb, text_emb.T)  # [B, B]

    # 训练开始
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        start_time = time()

        # tqdm 显示训练进度
        for images, texts, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", ncols=100):
            images = images.to(device)
            encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            image_emb, text_emb = model(images, input_ids, attention_mask)  # [B, D], [B, D]

            # 创建目标（假设这里是正样本）
            targets = torch.ones(image_emb.size(0)).to(device)

            # 计算对比损失
            loss = criterion(image_emb, text_emb, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算相似度矩阵
            sim_matrix = compute_similarity(image_emb, text_emb)  # [B, B]
            # 获取预测准确度
            correct = torch.argmax(sim_matrix, dim=1) == targets
            total_train_correct += correct.sum().item()
            total_train_samples += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples
        elapsed_time = time() - start_time

        print(
            f"📘 Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Train Accuracy: {train_accuracy * 100:.2f}% - Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

        # 使用 WandB 记录训练过程
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_loss,
            "Train Accuracy": train_accuracy,
            "Elapsed Time": elapsed_time
        })

        # 保存模型
        save_name = f"{SAVE_PATH}/dual_encoder_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), save_name)
        print(f"💾 Model saved at {save_name}")

    wandb.finish()  # 完成 WandB 记录
