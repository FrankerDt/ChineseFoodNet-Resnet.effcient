train_dataset = ChineseFoodPairDataset(
    image_root="./ChineseFoodNet/release_data/train/",
    list_txt_path="./ChineseFoodNet/release_data/train_list.txt",
    class_name_csv="./class_names.csv",
    recipe_json_path="./recipes_chinesefoodnet_207.json"
)
# 在训练脚本中调用时，传入你训练好的 `.pt` 权重路径
model = DualEncoderModel(embed_dim=256, image_weight_path="model_data/efficient_cbam/efficient_cbam-72-0.7427.pt")