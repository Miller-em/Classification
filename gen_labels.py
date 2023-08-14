import os
import csv

# 根据你自己的实际路径进行调整
train_dir = 'G:/Projects/Classfication/dogs-vs-cats/train'

# 获取文件夹下的所有图像文件
image_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]

# 创建csv文件
csvfile = "label.csv"
with open(csvfile, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Animal', 'Label'])

    # 遍历图像文件并写入CSV文件
    for image_file in image_files:
        image_path = os.path.join(train_dir, image_file)
        label = image_file.split('.')[0]  # 使用文件名作为分类
        id = 0 if label == 'cat' else 1
        writer.writerow([image_path, id])