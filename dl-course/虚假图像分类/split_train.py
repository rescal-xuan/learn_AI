import os
import random
import shutil

# 定义源文件夹路径和目标文件夹路径
train_dir = "./data/train"
valid_dir = "./data/valid"

# 创建目标文件夹
os.makedirs(valid_dir, exist_ok=True)

# 遍历源文件夹中的子文件夹
for folder in ["true", "fake"]:
    # 获取当前子文件夹的路径
    folder_path = os.path.join(train_dir, folder)
    # 在目标文件夹中创建对应的子文件夹
    valid_folder_path = os.path.join(valid_dir, folder)
    os.makedirs(valid_folder_path, exist_ok=True)
    
    # 获取当前子文件夹中的所有文件
    files = os.listdir(folder_path)
    # 计算需要移动的文件数量
    num_files = len(files)
    num_valid_files = int(0.3 * num_files)
    
    # 随机选择要移动的文件
    valid_files = random.sample(files, num_valid_files)
    
    # 移动文件到目标文件夹
    for file in valid_files:
        src = os.path.join(folder_path, file)
        dst = os.path.join(valid_folder_path, file)
        shutil.move(src, dst)
