from torch.utils.data import DataLoader
from ChineseFoodPairDataset import ChineseFoodPairDataset

if __name__ == "__main__":
    dataset = ChineseFoodPairDataset(
        image_root="./ChineseFoodNet/release_data/train/",
        list_txt_path="./ChineseFoodNet/release_data/train_list.txt",
        class_name_csv="./class_names.csv",
        recipe_json_path="./recipes_chinesefoodnet_207.json"
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # 打印前1个 batch
    for images, texts, labels in dataloader:
        print("✅ 图像 batch 维度:", images.shape)  # [B, 3, 224, 224]
        print("📖 文本样例:")
        for i, text in enumerate(texts):
            print(f"[{labels[i]}] {text[:50]}...")
        break
