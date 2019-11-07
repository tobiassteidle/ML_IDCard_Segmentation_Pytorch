import argparse
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import load_model

from utils import metrics

parser = argparse.ArgumentParser(description='Semantic segmentation of IDCard in Image.')
parser.add_argument('input', type=str, help='Image (with IDCard) Input file')
parser.add_argument('--output', type=str, default='prediction.png', help='Output file for image')
parser.add_argument('--model', type=str, default='model.h5', help='Path to .h5 model file')

args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output
MODEL_FILE = args.model

if not os.path.isfile(INPUT_FILE):
    print('Input image not found ', INPUT_FILE)

if not os.path.isfile(MODEL_FILE):
    print('Model not found ', MODEL_FILE)

print('Load image... ', INPUT_FILE)
img = cv2.imread(INPUT_FILE)
h, w = img.shape[:2]
img = cv2.resize(img, (256, 256))
#img = img / 255.0



img = img.reshape(1, 256, 256, 3)

print('Load model... ', MODEL_FILE)
model = load_model(MODEL_FILE,
                   custom_objects={'IoU': metrics.IoU})

print('Prediction...')
predict = model.predict(img, verbose=1)

print(predict.shape)

output_image = predict[0]
output_image = cv2.resize(output_image, (w,h))
plt.imsave(OUTPUT_FILE, output_image, cmap='gray')


cv2.imwrite('test/output_cv2.png', output_image)
