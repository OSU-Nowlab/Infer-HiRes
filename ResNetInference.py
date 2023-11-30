import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader

from utils import WSIDataloader

def load_dataset():
    batch_size = 1
    num_workers = 1

    transform = transforms.Compose(
        [ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    torch.manual_seed(0)

    # testset = torchvision.datasets.ImageNet(
    #         root="/home/gulhane.2/GEMS_Inference/datasets/ImageNet/", split='val', transform=transform
    # )
    testset = torchvision.datasets.ImageFolder(
        root="/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/single_gpu/imagenette2-320/val",
        transform=transform,
        target_transform=None,
    )

    dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
    )

    return dataloader


def load_wsi_dataset():
    batch_size = 1
    num_workers = 1

    # transform = transforms.Compose([
    #     transforms.Resize((tile_size, tile_size)),  # Adjust the size as needed
    #     transforms.ToTensor(),
    # ])
    testset = WSIDataloader.WSIDataloader(
        root="/home/gulhane.2/GEMS_Inference/datasets/test_digital_pathology/wsi_images",
        transform=None)

    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False)

    return dataloader

def load_torchResNet(device):
    CHECKPOINT_PATH = '/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth'
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet50()
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        print(param.requires_grad)
    exit()
    model.eval()
    model.to(device)
    return model

def load_torchResNetCustomClass(device, num_classes = 10):
    CHECKPOINT_PATH = '/home/gulhane.2/GEMS_Inference/checkpoints/torch_ResNet50/resnet50-19c8e357.pth'
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = resnet50()
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.eval()
    model.to(device)
    return model

def load_model(device, num_classes = 1000):
    if num_classes == 1000:
        return load_torchResNet(device)
    return load_torchResNetCustomClass(device, num_classes)
    #return load_torchResNet(device, classes)

# def tile_image(image, tile_size = 256):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(tile_size),
#         transforms.ToTensor()
#     ])
#     total_tiles_h = image.shape[1] // tile_size
#     total_tiles_w = image.shape[2] // tile_size

#     tiles = []
#     for i in range(total_tiles_h):
#         for j in range(total_tiles_w):
#             tile = image[:, i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
#             tile = transform(tile)
#             tiles.append(tile)
#     return torch.stack(tiles)

def evaluate_wsi(dataloader, model, device):
    import openslide

    corrects = 0
    t = time.time()
    perf = []
    count = 0

    # For now, considering batch-size of 1 when using tiling. Need to think of optimization.
    for wsi_imag_path, labels in dataloader:

        slide = openslide.OpenSlide(wsi_imag_path[0])
        tile_size = 256
        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        start_event.record()
        for x_pix in range(0, slide.dimensions[0], tile_size):
            for y_pix in range(0, slide.dimensions[1], tile_size):
                region = slide.read_region((x_pix, y_pix), 0, (tile_size, tile_size))
                pil_img = region.convert("RGB") # RGBA to RGB

                transform = transforms.ToTensor()
                tensor_image = transform(pil_img)

                batch_image = tensor_image.unsqueeze(dim=0)


                batch_image, labels = batch_image.to(device), labels.to(device)



                outputs = model(batch_image)


                _, predicted = torch.max(outputs, 1)
                corrects += (predicted == labels).sum().item()
                count += 1
        end_event.record()
        t = start_event.elapsed_time(end_event) / 1000
        perf.append(1 / t)
        print(f"Completed wsi_image")

        slide.close()
    accuracy = corrects / count #per slide accuracy  : need to rethink , we might look for per image accuracy
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


def evaluate_traditional(dataloader, model, device):
    corrects = 0
    t = time.time()
    perf = []

    with torch.no_grad():
        for batch, labels in dataloader:
            batch, labels = batch.to(device), labels.to(device)

            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            outputs = model(batch)

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            perf.append(1 / t)

            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
    accuracy = corrects / len(dataloader.dataset)
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    return accuracy

def evaluate(model, device, wsi_images = False):

    if wsi_images:
        dataloader = load_wsi_dataset()
        accuracy = evaluate_wsi(dataloader, model, device)
    else:
        dataloader = load_dataset()
        accuracy = evaluate_traditional(dataloader, model, device)
    return accuracy

device = "cuda:0"
wsi_images = False
num_classes = 10

model = load_model(device, num_classes)
accuracy = evaluate(model, device, wsi_images)
print(f"Accuracy with pretrained model : {accuracy * 100}")