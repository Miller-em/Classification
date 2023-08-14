import torch
import torchvision.models
import torch.nn as nn

from PIL import Image

from utils import get_transforms


def predict():
    device = torch.device("cuda")
    model_path = r"vit_best.pth"
    checkpoint = torch.load(model_path)

    model = torchvision.models.vit_b_16()
    model.heads = nn.Sequential(nn.Linear(768, 2))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    inference_image = r"dogs-vs-cats\test\209.jpg"
    image = Image.open(inference_image)

    transform = get_transforms()
    image = transform(image).to(device)
    image = image.unsqueeze(0)

    # inference
    output = model(image)
    _, prediction = output.max(dim=1)
    if prediction[0] == 0:
        print("Cat")
    else:
        print("Dog")


if __name__ == "__main__":
    predict()