import argparse
import cv2
import numpy as np
import os
import pathlib

import torch

import models
from utils import image

parser = argparse.ArgumentParser(description='Semantic segmentation of IDCard in Image.')
parser.add_argument('input', type=str, help='Image (with IDCard) Input file')
parser.add_argument('--output_mask', type=str, default='output_mask.png', help='Output file for mask')
parser.add_argument('--output_prediction', type=str, default='output_pred.png', help='Output file for image')
parser.add_argument('--model', type=str, default='./pretrained/model_checkpoint.pt', help='Path to checkpoint file')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_FILE = args.input
OUTPUT_MASK = args.output_mask
OUTPUT_FILE = args.output_prediction
MODEL_FILE = args.model


def predict_image(model, image):
    with torch.no_grad():
        output = model(image.to(device))

    output = output.detach().cpu().numpy()[0]
    output = output.transpose((1, 2, 0))
    output = np.uint8(output)
    _, output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY_INV)

    return output


def main():
    if not os.path.isfile(INPUT_FILE):
        print('Input image not found ', INPUT_FILE)
    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)
            model = models.UNet(n_channels=1, n_classes=1)

            checkpoint = torch.load(pathlib.Path(MODEL_FILE))
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            print('Load image... ', INPUT_FILE)
            img, h, w = image.load_image(INPUT_FILE)

            print('Prediction...')
            output_image = predict_image(model, img)

            print('Resize mask to original size...')
            mask_image = cv2.resize(output_image, (w, h))
            cv2.imwrite(OUTPUT_MASK, mask_image)

            print('Cut it out...')
            warped = image.extract_idcard(cv2.imread(INPUT_FILE), mask_image)
            cv2.imwrite(OUTPUT_FILE, warped)

            print('Done.')


if __name__ == '__main__':
    main()
