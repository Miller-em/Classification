import argparse
import torchvision.transforms as transforms
import logging
import time
import os
import torch

def get_parse():
    """
        配置各种参数
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--learning_rate', type=int, default=0.001, help="learning rate")
    parser.add_argument('--n_epoches', type=int, default=20, help="the total number of training epoches")
    parser.add_argument('--pretrain', type=bool, default=True, help="whether use pretrained model")
    parser.add_argument('--resume_from', type=bool, default=False, help="resume training")
    parser.add_argument('--param_frozen', type=bool, default=True,
                        help="freeze parameters outside of fully-connected layers in the pretrained model")
    parser.add_argument('--save_log_dir', type=str, default="Logs/vit", help="the directory path of logfile")
    parser.add_argument('--eval_iterval', type=int, default=1, help="evaluate model intervals")
    args = parser.parse_args()
    return args

def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4881, 0.4547, 0.4167], std=[0.2595, 0.2528, 0.2554]
        ), # this is calculated by get_mean_std
    ])
    return transform

def get_logger(root_dir, mode="w"):
    # step 1: create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # step 2: create a handler for writing to log file
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    root_dir = os.path.join(os.getcwd(), root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    log_file = os.path.join(root_dir, rq + '.log')
    fh = logging.FileHandler(log_file, mode=mode)
    fh.setLevel(logging.DEBUG)

    # Step 3: Define the output format of handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # step 4: Add logger to handler
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def saveModel(logger, model, optimizer, scheduler, epoch, best_acc, filename="best_model.pth"):
    logger.info("Starting to save the best model...............")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc
    }
    torch.save(checkpoint, os.path.join(os.getcwd(), filename))

if __name__ == "__main__":
    args = get_parse()
    logger = get_logger(args.save_log_dir)
    logger.info("Start...................")