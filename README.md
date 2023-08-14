# Image Classification using Pytorch

This repo is an image classification implementation from scratch. It includes models, custom dataset class, 
train&evaluate pipeline and predict.

## 1. Supported Models
We support the following classification algorithms (models) :

- VGG16
- ResNet (layer50, layer101, layer152)
- Vision Transformer (ViT)

Place the specific code implementation in the **"models/"** folder.

## 2. Dataset
1. We are using cats&dogs classification datasets, you download it from here: [link](https://pan.baidu.com/s/1_mvcB0Il63SKKF5MTBVt5w?pwd=pm07)
, and you place it to this project.
2. Modify the "train_dir" path in the "gen_label.py" file. This path should point to the directory of your dataset's training set.
3. Here we use csv file to record the datasets labels. The cat-class is assigned with 0 label, and the dog-class is 1 label.
we use the following commands to generate csv label file:
```bash
python gen_label.py
```

## 3. How to use 
1. Train


2. Predict