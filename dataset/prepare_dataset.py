import cv2
import json
import numpy as np
import os
import wget
import zipfile
import shutil
from glob import glob

download_links = [
    'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
    'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
    'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
    'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
    'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
    'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip']

PATH_OFFSET = 40
TARGET_PATH = 'dataset/data/'

RESULT_PATH = 'dataset/images/'
IMAGE_PATH = RESULT_PATH + 'image/'
MASK_PATH = RESULT_PATH + 'mask/'

def read_image(img, label):
    image = cv2.imread(img)
    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.resize(mask, (256, 256))
    image = cv2.resize(image, (256, 256))
    return image, mask

def main():
    # Remove Temp Directory and create a new one
    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH, ignore_errors=True)

    os.mkdir(RESULT_PATH)
    os.mkdir(IMAGE_PATH)
    os.mkdir(MASK_PATH)

    # Counter for filename
    file_idx = 1

    for link in download_links:
        filename = link[PATH_OFFSET:]
        full_filename = TARGET_PATH + filename
        directory_name = TARGET_PATH + link[PATH_OFFSET:-4]

        print('Collect and prepare datasets...')

        print('Dataset available... ', directory_name)
        if not os.path.exists(directory_name):
            if not os.path.isfile(full_filename):
                # file not found, execute wget download
                print ('Downloading:', link)
                wget.download(link, TARGET_PATH)

            else:
                # Unzip archives
                with zipfile.ZipFile(full_filename, 'r') as zip_ref:
                    zip_ref.extractall(TARGET_PATH)

        print('Prepare dataset... ', directory_name)
        img_dir_path = './' + directory_name + '/images/'
        gt_dir_path = './' + directory_name + '/ground_truth/'

        # Remove unessesary files
        if os.path.isfile(img_dir_path + filename + '.tif'):
            os.remove(img_dir_path + filename.replace('.zip', '.tif'))
        if os.path.isfile(gt_dir_path + filename + '.json'):
            os.remove(gt_dir_path + filename.replace('.zip', '.json'))

        # Load Images and Groundtruth and store as numpy array
        for images, ground_truth in zip(sorted(os.listdir(img_dir_path)), sorted(os.listdir(gt_dir_path))):
            img_list = sorted(glob(img_dir_path + images + '/*.tif'))
            label_list = sorted(glob(gt_dir_path + ground_truth + '/*.json'))
            for img, label in zip(img_list, label_list):
                image, mask = read_image(img, label)
                cv2.imwrite(IMAGE_PATH + 'image' + str(file_idx) + '.png', image)
                cv2.imwrite(MASK_PATH + 'image' + str(file_idx) + '.png', mask)

                file_idx += 1

        print('----------------------------------------------------------------------')

    # Done

if __name__ == '__main__':
    main()
