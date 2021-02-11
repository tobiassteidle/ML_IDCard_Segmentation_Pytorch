# https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py

import cv2
import numpy as np
from PIL import Image
import torch


def load_image(input_file):
    source_img = Image.open(input_file).convert('L')

    # start with an quadratic image
    offset = 20
    size = np.max([source_img.size, source_img.size])
    image = Image.new('L', (size + offset, size + offset))
    image.paste(source_img, (offset // 2, offset // 2))

    width, height = image.size

    # resize for inference
    image = image.resize((256, 256))
    img_nd = np.array(image)

    # expand grayscale image to 3 dimensions
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans / 255

    # reshape to 1-batched tensor
    img_trans = img_trans.reshape(1, 1, 256, 256)

    return torch.from_numpy(img_trans).type(torch.FloatTensor), height, width


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_contours(image, thickness=3):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness)
    return contour_image, contours, hierarchy


def extract_idcard(raw_image, mask_image):
    contour_image, contours, hierarchy = find_contours(mask_image)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx

        if (len(screenCnt) == 4):
            screenCntList.append(screenCnt)

    assert len(screenCntList) == 1
    new_points = np.array([[points[0][0], points[0][1]] for points in screenCntList[0]])

    warped = four_point_transform(raw_image, new_points)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
