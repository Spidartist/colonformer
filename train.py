import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    
epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(train_loader, start=1):
            if epoch <= 1:
                    optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates: 
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                map4, map3, map2, map1 = model(images)
                map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
                # with torch.autograd.set_detect_anomaly(True):
                #loss = nn.functional.binary_cross_entropy(map1, gts)
                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

            # ---- train visualization ----
            if i == total_step:
                print('{} Training Epoch [{:03d}/{:03d}], '
                        '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                        format(datetime.now(), epoch, args.num_epochs,\
                                loss_record.show(), dice.show(), iou.show()))

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='ConlonFormerB3')
    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='')
    args = parser.parse_args()

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    class Benchmark(Dataset):
        def __init__(
                self, 
                path="/mnt/quanhd/endoscopy/public_dataset.json", 
                mode="train",
                ds_test="CVC-300",
                img_size=256,
                root_path="home/s/DATA/",
                train_ratio=1.0
                # root_path="/mnt/tuyenld/data/endoscopy/"

            ):
            self.path = path
            self.img_size = img_size
            self.mode = mode
            self.ds_test = ds_test
            self.train_ratio = train_ratio
            self.root_path = root_path
            self.load_data_from_json()

        def load_data_from_json(self):
            with open(self.path) as f:
                data = json.load(f)
            if self.mode == "train":
                all_image_paths = data[self.mode]["images"]
                kvasir_image_paths = []
                clinic_image_paths = []
                for image_path in all_image_paths:
                    if "c" in image_path:
                        kvasir_image_paths.append(image_path)
                    else:
                        clinic_image_paths.append(image_path)
            
                all_mask_paths = data[self.mode]["masks"]
                kvasir_mask_paths = []
                clinic_mask_paths = []
                for mask_path in all_mask_paths:
                    if "c" in mask_path:
                        kvasir_mask_paths.append(mask_path)
                    else:
                        clinic_mask_paths.append(mask_path)
                print(f"Pre len(all_image_paths) = {len(all_image_paths)}")
                print(f"Pre len(all_mask_paths) = {len(all_mask_paths)}")
                kvasir_image_paths[:int(len(kvasir_image_paths)*self.train_ratio)].extend(clinic_image_paths[:int(len(clinic_image_paths)*self.train_ratio)])
                kvasir_mask_paths[:int(len(kvasir_mask_paths)*self.train_ratio)].extend(clinic_mask_paths[:int(len(clinic_mask_paths)*self.train_ratio)])
                self.image_paths = kvasir_image_paths
                self.mask_paths = kvasir_mask_paths
                print(f"After len(image_paths) = {len(self.image_paths)}")
                print(f"After len(mask_paths) = {len(self.mask_paths)}")
            elif self.mode == "test":
                self.image_paths = data[self.mode][self.ds_test]["images"]
                self.mask_paths = data[self.mode][self.ds_test]["masks"]
        
        def aug(self, image, mask):
            if self.mode == 'train':
                t1 = A.Compose([A.Resize(self.img_size, self.img_size),])
                resized = t1(image=image, mask=mask)
                image = resized['image']
                mask = resized['mask']

                t = A.Compose([                
                    A.HorizontalFlip(p=0.7),
                    A.VerticalFlip(p=0.7),
                    A.Rotate(interpolation=cv2.BORDER_CONSTANT, p=0.7),
                    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
                    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=0, p=0.35),
                    A.MotionBlur(p=0.2),
                    A.HueSaturationValue(p=0.2),                
                ], p=0.5)

            elif self.mode == 'test':
                t = A.Compose([
                    A.Resize(self.img_size, self.img_size)
                ])

            return t(image=image, mask=mask)
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, index):
            full_image_path = os.path.join(self.root_path, self.image_paths[index])
            full_mask_path = os.path.join(self.root_path, self.mask_paths[index])

            img = cv2.imread(full_image_path).astype(np.float32)
            orin_mask = cv2.imread(full_mask_path).astype(np.float32)

            augmented = self.aug(img, orin_mask)
            img = augmented['image']
            orin_mask = augmented['mask']

            img = torch.from_numpy(img.copy())
            img = img.permute(2, 0, 1)
            img /= 255.

            orin_mask = torch.from_numpy(orin_mask.copy())
            orin_mask = orin_mask.permute(2, 0, 1)
            orin_mask = orin_mask.mean(dim=0, keepdim=True)/255.
            orin_mask[orin_mask <= 0.5] = 0
            orin_mask[orin_mask > 0.5] = 1

            return img, orin_mask

    path = "/root/quanhd/endoscopy/public_dataset.json"
    train_dataset = Benchmark(path=path, root_path="/root/quanhd/DATA", img_size=args.init_trainsize, train_ratio=1.0)
            
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    model = UNet(backbone=dict(
                    type='mit_{}'.format(args.backbone),
                    style='pytorch'), 
                decode_head=dict(
                    type='UPerHead',
                    in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    decoder_params=dict(embed_dim=768),
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
                pretrained='/root/quanhd/ColonFormer/pretrained/mit_b2.bin'.format(args.backbone)).cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("#"*20, "Start Training", "#"*20)
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)