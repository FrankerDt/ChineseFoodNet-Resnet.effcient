# path_config.py
import os

# 获取当前这个文件所在目录（即项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录（如 ./ChineseFoodNet）
CHINESEFOODNET_ROOT = os.path.join(PROJECT_ROOT, 'ChineseFoodNet')

# 训练数据列表路径
TRAIN_LIST_PATH = os.path.join(CHINESEFOODNET_ROOT, 'release_data', 'train_list.txt')
VAL_LIST_PATH = os.path.join(CHINESEFOODNET_ROOT, 'release_data', 'val_list.txt')
TEST_LIST_PATH = os.path.join(CHINESEFOODNET_ROOT, 'release_data', 'test_list.txt')
