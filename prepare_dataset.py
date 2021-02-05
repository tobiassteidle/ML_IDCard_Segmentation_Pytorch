import cv2
import json
import numpy as np
import os
import random
import re
import shutil
import wget
import zipfile
from PIL import Image
from glob import glob
'''
download_links = [
    'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
    'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
    'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
    'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
    'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
    'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip']
'''

download_links = ['ftp://smartengines.com/midv-500/dataset/01_alb_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/04_aut_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/07_chl_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip',
                  'ftp://smartengines.com/midv-500/dataset/09_chn_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/10_cze_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/22_est_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/24_fin_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/33_mac_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/37_prt_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip',
                  'ftp://smartengines.com/midv-500/dataset/40_srb_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/42_svk_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/43_tur_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip',
                  'ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip',
                  'ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip',
                  'ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip']

PATH_OFFSET = 40
TARGET_PATH = 'dataset/data/'

TEMP_PATH = 'dataset/temp/'
TEMP_IMAGE_PATH = TEMP_PATH + 'image/'
TEMP_MASK_PATH = TEMP_PATH + 'mask/'

DATA_PATH = 'dataset/train/'

SEED = 230


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
    return image, mask


def download_and_unzip():
    # Remove Temp Directory and create a new one
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH, ignore_errors=True)

    os.mkdir(TEMP_PATH)
    os.mkdir(TEMP_IMAGE_PATH)
    os.mkdir(TEMP_MASK_PATH)

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
                cv2.imwrite(TEMP_IMAGE_PATH + 'image' + str(file_idx) + '.png', image)
                cv2.imwrite(TEMP_MASK_PATH + 'image' + str(file_idx) + '.png', mask)

                file_idx += 1

        print('----------------------------------------------------------------------')


def train_validation_split():
    # Remove Temp Directory and create a new one
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH, ignore_errors=True)

    # Create folders to hold images and masks
    folders = ['train_frames/image', 'train_masks/image', 'val_frames/image', 'val_masks/image', 'test_frames/image',
               'test_masks/image']

    for folder in folders:
        os.makedirs(DATA_PATH + folder)

    # Get all frames and masks, sort them, shuffle them to generate data sets.
    all_frames = os.listdir(TEMP_IMAGE_PATH)
    all_masks = os.listdir(TEMP_MASK_PATH)

    all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])

    random.seed(SEED)
    random.shuffle(all_frames)

    # Generate train, val, and test sets for frames
    train_split = int(0.7 * len(all_frames))
    val_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[val_split:]

    # Generate corresponding mask lists for masks
    train_masks = [f for f in all_masks if f in train_frames]
    val_masks = [f for f in all_masks if f in val_frames]
    test_masks = [f for f in all_masks if f in test_frames]

    # Add train, val, test frames and masks to relevant folders
    def add_frames(dir_name, image):
        img = Image.open(TEMP_IMAGE_PATH + image)
        img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)

    def add_masks(dir_name, image):
        img = Image.open(TEMP_MASK_PATH + image)
        img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)

    frame_folders = [(train_frames, 'train_frames/image'), (val_frames, 'val_frames/image'),
                     (test_frames, 'test_frames/image')]
    mask_folders = [(train_masks, 'train_masks/image'), (val_masks, 'val_masks/image'),
                    (test_masks, 'test_masks/image')]

    print('Split images into train, test and validation...')

    # Add frames
    for folder in frame_folders:
        array = folder[0]
        name = [folder[1]] * len(array)
        list(map(add_frames, name, array))

    # Add masks
    for folder in mask_folders:
        array = folder[0]
        name = [folder[1]] * len(array)
        list(map(add_masks, name, array))


def main():
    download_and_unzip()

    train_validation_split()


if __name__ == '__main__':
    main()
