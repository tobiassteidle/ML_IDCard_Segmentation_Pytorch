#https://github.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/blob/master/four_point_object_extractor.py

import cv2
import numpy as np

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


def findLargestCountours(cntList, cntWidths):
    newCntList = []
    newCntWidths = []

    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)
    return newCntList, newCntWidths


def convert_object(mask, image):
    gray = mask
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.medianBlur(gray, 5)
    edged = cv2.Canny(gray, 30, 400)
    countours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cnts = sorted(countours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    scrWidths = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx

        if (len(screenCnt) == 4):
            (X, Y, W, H) = cv2.boundingRect(cnt)
            screenCntList.append(screenCnt)
            scrWidths.append(W)

    if len(scrWidths) != 2:
        print('ID Card not found.')
        return None
    else:
        screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)

        if not len(screenCntList) >= 2:  # there is no rectangle found
            return None
        elif scrWidths[0] != scrWidths[1]:  # mismatch in rect
            return None

        pts = screenCntList[0].reshape(4, 2)
        warped = four_point_transform(image, pts)
        return warped
