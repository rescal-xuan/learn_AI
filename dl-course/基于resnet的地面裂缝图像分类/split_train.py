import os
import shutil
import random

# 设置训练集和验证集的根目录
train_dir = "./train"
valid_dir = "./valid"

# 设置验证集的比例
valid_ratio = 0.2

# 遍历所有类别文件夹
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        # 创建验证集类别文件夹
        valid_class_path = os.path.join(valid_dir, class_name)
        os.makedirs(valid_class_path, exist_ok=True)

        # 获取类别文件夹中的所有图片文件名
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 计算要移动到验证集的图片数量
        num_valid_images = int(len(image_files) * valid_ratio)

        # 随机选择图片移动到验证集
        random.shuffle(image_files)
        valid_images = image_files[:num_valid_images]

        # 移动图片到验证集文件夹
        for image_file in valid_images:
            source_path = os.path.join(class_path, image_file)
            destination_path = os.path.join(valid_class_path, image_file)
            shutil.move(source_path, destination_path)

print("图片已成功移动到验证集文件夹！")