import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
from pathlib import Path


class ChineseFoodPairDataset(Dataset):
    def __init__(self,
                 image_root: str,
                 list_txt_path: str,
                 class_name_csv: str,
                 recipe_json_path: str,
                 transform=None):
        self.image_root = image_root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785],
                                 std=[0.25595631, 0.25862494, 0.26925405])
        ])

        # 读取图像路径和类别编号
        with open(list_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                rel_path, class_id = parts
                class_id = int(class_id)
                self.samples.append((rel_path, class_id))

        # 加载 class_names
        class_df = pd.read_csv(class_name_csv, header=None, encoding='gbk')
        self.class_names = class_df[1].tolist()  # 中文菜名

        # 加载菜谱文本
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            recipes_raw = json.load(f)
            recipes = recipes_raw[0] if isinstance(recipes_raw, list) and isinstance(recipes_raw[0],list) else recipes_raw
        self.label2text = {}
        for entry in recipes:
            label = entry['label']
            ing = '、'.join(entry['ingredients'])
            steps = ' '.join(entry['steps'])
            self.label2text[label] = f"{ing}。{steps}"

    def __getitem__(self, index):
        rel_path, class_id = self.samples[index]
        img_path = Path(self.image_root) / rel_path
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label_name = self.class_names[class_id]
        text = self.label2text.get(label_name, "无对应菜谱。")
        return image, text, class_id

    def __len__(self):
        return len(self.samples)
