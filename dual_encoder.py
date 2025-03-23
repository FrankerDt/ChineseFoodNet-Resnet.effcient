
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel
from model.efficient_cbam import EfficientCBAMResNet


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, weight_path="model_data/efficient_cbam/efficient_cbam-72-0.7427.pt", device=None):
        super(ImageEncoder, self).__init__()

        # 使用自定义的 EfficientCBAMResNet 模型
        self.backbone = EfficientCBAMResNet(num_classes=embed_dim)  # 使用 embed_dim 输出大小

        # 如果有权重文件，加载模型权重
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location='cpu')  # 先加载到 CPU
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功加载自定义图像塔权重: {weight_path}")

        # 移动模型到正确的设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(self.device)  # 确保模型加载到当前设备（GPU 或 CPU）

    def forward(self, images):
        images = images.to(self.device)  # 确保输入数据在正确的设备上
        x = self.backbone(images)  # [B, embed_dim]
        x = F.normalize(x, p=2, dim=1)  # 单位化向量
        return x

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained_model='bert-base-chinese'):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.fc(pooled)
        x = F.normalize(x, p=2, dim=1)
        return x


class DualEncoderModel(nn.Module):
    def __init__(self, embed_dim=256, image_weight_path="model_data/resmodel-41-4000-0.9642.pt"):
        super(DualEncoderModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim, weight_path=image_weight_path)
        self.text_encoder = TextEncoder(embed_dim)

    def forward(self, images, input_ids, attention_mask):
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(input_ids, attention_mask)
        return image_emb, text_emb
