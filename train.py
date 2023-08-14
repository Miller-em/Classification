import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CatandDogDataset
from models.vit import ViT
from utils import get_parse, get_transforms, get_logger, saveModel
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(logger, epoch, n_epochs, model, criterion, optimizer, scheduler, train_loader, writer, device):
    model.train()
    total_step = len(train_loader)
    losses = []
    running_loss = 0.0
    step = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)
        running_loss += loss.item()
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Draw
        writer.add_scalar("Training Loss", loss.item(), global_step=step)
        step += 1

        if (batch_idx+1) % 100 == 0:
            logger.info(f"Epoch [{epoch}/{n_epochs}], Step [{batch_idx+1}/{total_step}], LR = {optimizer.state_dict()['param_groups'][0]['lr']}, Loss = {running_loss/100:.4f}")
            running_loss = 0.0

    mean_loss = sum(losses)/len(losses)
    mean_loss = round(mean_loss, 2)
    logger.info(f"mean_loss: {mean_loss}")
    scheduler.step(mean_loss)


@torch.no_grad()
def evaluate(epoch, logger, model, test_loader, writer, device):
    logger.info('Stating to evaluate .........................')
    model.eval()
    num_correct = 0
    num_total = 0

    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        scores = model(X)
        _, predictions = scores.max(dim=1)
        num_correct += (y == predictions).sum()
        num_total += X.shape[0]

    logger.info(f"Got {num_correct}/{num_total} with accuracy {float(num_correct) / float(num_total) * 100:.2f}")
    eval_acc = float(num_correct) / float(num_total)
    writer.add_scalar("Evaluate Accuracy", eval_acc, global_step=epoch)  # write to tensorboard
    return eval_acc

def main(args):
    # define hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = 1
    best_acc = 0.0

    # define model
    if args.pretrain:
        model = torchvision.models.vit_b_16(pretrained=args.pretrain)
        if args.param_frozen:
            for param in model.parameters():
                param.requires_grad = False
        model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=2)
        )
    else:
        model = ViT()  # all use default parameters
    model.to(device)

    # Load Data
    transform = get_transforms()
    dataset = CatandDogDataset("label.csv", transform=transform)
    train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Define Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    # Resume model from got path
    if args.resume_from:
        checkpoint = torch.load("vit_best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    # get logger
    logger = get_logger(args.save_log_dir)

    # writer
    writer = SummaryWriter(f"runs/dogs-vs-cats/")

    for epoch in range(start_epoch, args.n_epoches+1):
        # train
        train_one_epoch(logger, epoch, args.n_epoches, model, criterion, optimizer, scheduler, train_loader, writer, device)

        # eval
        if epoch % args.eval_iterval == 0:
            acc = evaluate(epoch, logger, model, test_loader, writer, device)
            if acc > best_acc:
                best_acc = acc
                saveModel(logger, model, optimizer, scheduler, epoch, best_acc, "vit_best.pth")
                logger.info(f"saved the best model with accuracy {acc * 100:.2f}")
    logger.info("Training Finished!!!!!!!!!")

if __name__ == "__main__":
    args = get_parse()
    main(args)