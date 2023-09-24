import os
import torch
import torchvision
import torch.nn as nn
import timm
from timm.utils import accuracy
import math
import glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    for batch in data_loader:
        images = batch[0]
        targe = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        output = torch.nn.functional.softmax(output, dim=1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

def build_transform(is_train, args):
    if is_train:
        print("train transform")
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(args.input_size, args.input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                torchvision.transforms.ToTensor()
            ]
        )

    print("eval transform")
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor()
        ]
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    return dataset


def main(args, mode='train', test_image_path=''):
    if mode == 'train':
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        model = timm.create_model('resnet18', pretrained=True, num_classes=36, drop_rate=0.1, drop_path_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        os.makedirs(args.log_dir, exist_ok=True)

        log_writer = SummaryWriter(log_dir=args.log_dir)

        loss_scaler = NativeScaler()

        for epoch in range(args.start_epoch, args.epoch):
            if epoch % 1 == 0:
                print("Evaluating...")
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)

                if log_writer is not None:
                    log_writer.add_scalar('pref/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('pref/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('pref/test_loss', test_stats['loss'], epoch)
                model.train()

                print("Training...")


                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                    )