import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch import nn, optim
from tqdm import tqdm
import os

from dual_encoder import DualEncoderModel
from ChineseFoodPairDataset import ChineseFoodPairDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 参数设置
BATCH_SIZE = 32
EPOCHS = 10
EMBED_DIM = 256
LR = 1e-4
SAVE_PATH = "./dual_model"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# ✅ 加载图文配对数据集
train_dataset = ChineseFoodPairDataset(
    image_root="./ChineseFoodNet/release_data/train/",
    list_txt_path="./ChineseFoodNet/train_list.txt",
    class_name_csv="./ChineseFoodNet/class_names.csv",
    recipe_json_path="./ChineseFoodNet/recipes_chinesefoodnet_207_cleaned.json"
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ✅ 模型初始化（默认加载图像塔预训练 .pt 文件）
model = DualEncoderModel(embed_dim=EMBED_DIM).to(device)

# ✅ 损失函数：简化版 InfoNCE（对角最大化）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def compute_similarity(image_emb, text_emb):
    """计算图文之间的余弦相似度矩阵"""
    return torch.matmul(image_emb, text_emb.T)  # [B, B]

# ✅ 开始训练
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, texts, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # 文本 → BERT Tokenizer 编码
        encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        images = images.to(device)
        image_emb, text_emb = model(images, input_ids, attention_mask)  # [B, D], [B, D]

        sim_matrix = compute_similarity(image_emb, text_emb)  # [B, B]
        targets = torch.arange(sim_matrix.size(0)).to(device)

        loss = criterion(sim_matrix, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📘 Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    # ✅ 保存模型
    save_name = f"{SAVE_PATH}/dual_encoder_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), save_name)
    print(f"💾 模型已保存至 {save_name}")
