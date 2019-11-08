import argparse
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import load_model

parser = argparse.ArgumentParser(description='Semantic segmentation of IDCard in Image.')
parser.add_argument('input', type=str, help='Image (with IDCard) Input file')
parser.add_argument('--output', type=str, default='prediction.png', help='Output file for image')
parser.add_argument('--model', type=str, default='model.h5', help='Path to .h5 model file')

args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
MODEL_FILE = args.model


def load_image():
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    height, width = img.shape[:2]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 256, 256, 1)
    return img, height, width


def predict_image(model, image):
    predict = model.predict(image, verbose=1)
    return predict[0]


def main():
    if not os.path.isfile(INPUT_FILE):
        print('Input image not found ', INPUT_FILE)
    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)
            model = load_model(MODEL_FILE)

            print('Load image... ', INPUT_FILE)
            img, h, w = load_image()

            print('Prediction...')
            output_image = predict_image(model, img)

            print('Save output file...', OUTPUT_FILE)
            output_image = cv2.resize(output_image, (w, h))
            plt.imsave(OUTPUT_FILE, output_image, cmap='gray')

            print('Done.')


if __name__ == '__main__':
    main()
