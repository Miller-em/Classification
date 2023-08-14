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

We can train our model using this command:
```bash
python train.py --batch_size 16 --n_epoches 20 --pretrain False --save_log_dir "Logs/vit_test"
```
you know, options: **batch_size** is dataloader's batch size; **n_epoches** is the number of training epoches;
**pretrain** is False means not using a pre-trained model, while setting it to True means using a pre-trained model.
**save_log_dir** is the dir path of save log file.

Additionally, if you want to use the checkpoint resume function, use the following command:
```bash
python train.py --resume_from True
```


2. Predict

If you want to use a trained model to predict images, please read the "predict.py" file and modify the paths for the model
file and the image you want to predict. After that, use the following command:
```bash
python predict.py
```
