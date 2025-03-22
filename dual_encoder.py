
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, weight_path=None):
        super(ImageEncoder, self).__init__()
        base_model = models.resnet50(pretrained=(weight_path is None))
        modules = list(base_model.children())[:-1]  # 去除最后的全连接层
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, embed_dim)

        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功加载自定义图像塔权重: {weight_path}")

    def forward(self, images):
        x = self.backbone(images)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
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
