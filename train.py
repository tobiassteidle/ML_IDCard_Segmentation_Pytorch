import sys
import os
import numpy as np
import time
import argparse
import random
from os import walk
import pathlib
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from utils.metrics import multi_acc, iou_score

NO_OF_EPOCHS = 500
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)

SEED = 230

CHECKPOINT_PATH = pathlib.Path("./pretrained/model_checkpoint.pt")
FINAL_PATH = pathlib.Path("./pretrained/model_final.pt")

parser = argparse.ArgumentParser(description='Training Semantic segmentation of IDCard in Image.')
parser.add_argument('--resumeTraining', type=bool, default=False, help='Resume Training')

args = parser.parse_args()
RESUME_TRAINING = args.resumeTraining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


class SegmentationImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        _, _, self.filenames = next(walk(image_dir))

    @classmethod
    def preprocess(cls, pil_img, normalize=True):
        pil_img = pil_img.convert('L')

        pil_img = pil_img.resize(IMAGE_SIZE)
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if normalize:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + '/' + self.filenames[idx])
        mask = Image.open(self.mask_dir + '/' + self.filenames[idx])

        image = self.preprocess(image)
        mask = self.preprocess(mask)

        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return image, mask

    def __len__(self):
        return len(self.filenames)


def saveCheckpoint(filename, epoch, model, optimizer, batchsize):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "batch_size": batchsize,
    }

    # save all important stuff
    torch.save(checkpoint, filename)


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5, epochs_earlystopping=10):
    logdir = './logs/' + time.strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(logdir)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=logdir)

    best_acc = 0.0
    best_loss = sys.float_info.max
    best_iou = 0.0

    early_stopping = epochs_earlystopping

    for epoch in range(num_epochs):
        result = []
        early_stopping += 1

        for phase in ['train', 'val']:
            if phase == 'train':  # put the model in training mode
                model.train()
            else:
                # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            batch_nums = 0
            running_loss = 0.0
            running_iou = 0.0
            running_corrects = 0.0

            for (data, labels) in data_loader[phase]:
                # load the data and target to respective device
                (data, labels) = (data.to(device), labels.to(device))

                with torch.set_grad_enabled(phase == 'train'):
                    # feed the input
                    output = model(data)

                    # calculate the loss
                    loss = criterion(output, labels)

                    if phase == 'train':
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()

                        optimizer.step()

                        # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                # statistics
                batch_nums += 1
                running_loss += loss.item()
                running_iou += iou_score(output, labels)
                running_corrects += multi_acc(output, labels)

            if phase == 'train':
                scheduler.step(running_iou)

            # epoch statistics
            epoch_loss = running_loss / batch_nums
            epoch_iou = running_iou / batch_nums
            epoch_acc = running_corrects / batch_nums

            result.append('{} Loss: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_iou))

            tb_writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
            tb_writer.add_scalar('IoU/' + phase, epoch_iou, epoch)
            tb_writer.add_scalar('Accuracy/' + phase, epoch_acc, epoch)

            if phase == 'val' and epoch_iou > best_iou:
                early_stopping = 0

                best_acc = epoch_acc
                best_loss = epoch_loss
                best_iou = epoch_iou
                saveCheckpoint(CHECKPOINT_PATH, epoch, model, optimizer, BATCH_SIZE)
                print(
                    'Checkpoint saved - Loss: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(epoch_loss, epoch_acc, epoch_iou))

        print(result)

        if early_stopping == 10:
            break

    print('-----------------------------------------')
    print('Final Result: Loss: {:.4f} Acc: {:.4f}'.format(best_loss, best_acc))
    print('-----------------------------------------')


def main():
    seed_torch()

    print('Create datasets...')
    train_dataset = SegmentationImageDataset('./dataset/train/train_frames/image', './dataset/train/train_masks/image')
    validation_dataset = SegmentationImageDataset('./dataset/train/val_frames/image', './dataset/train/val_masks/image')

    print('Create dataloader...')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    dataloader = {"train": train_dataloader,
                  "val": validation_dataloader}

    print('Initialize model...')
    model = models.UNet(n_channels=1, n_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    print(RESUME_TRAINING)
    if RESUME_TRAINING:
        print('Load Model to resume training...')
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    print('Start training...')
    train(model, dataloader, criterion, optimizer, scheduler, num_epochs=NO_OF_EPOCHS)

    print('Save final model...')
    torch.save(model.state_dict(), FINAL_PATH)


if __name__ == '__main__':
    main()
