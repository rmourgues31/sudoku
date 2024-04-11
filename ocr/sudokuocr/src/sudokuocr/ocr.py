import cv2
import numpy as np

import easyocr
reader = easyocr.Reader(['en'])

def get_perspective(img, location, height = 900, width = 900):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh

def parse_image(img):

# Read image
    saturated = increase_brightness(img)
    blue = contrast(saturated)

    contours, hierarchy = cv2.findContours(blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    newimg = cv2.drawContours(saturated.copy(), contours, -1, (0, 255, 0), 3)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    # Finds rectangular contour
    for i,contour in enumerate(contours):
        epsilon = 0.1*cv2.arcLength(contour,True)
        newimg = cv2.drawContours(img.copy(), contour, -1, (0, 255, 0), 3)

        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            location = approx
            break
    location = np.roll(location, -np.argmin(location.sum(axis=1)), axis=0)
    result = get_perspective(img, location)

    def split_boxes2(board):
        """Takes a sudoku board and split it into 81 cells. 
            each cell contains an element of that board either given or an empty cell."""
        rows = np.vsplit(board,9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)
            for box in cols:
                boxes.append(box)
        return boxes

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    rois = split_boxes2(gray)

    m = []

    for i,r in enumerate(rois):
        predicted = reader.readtext(r, allowlist='0123456789')
        m.append(int(predicted[0][-2]) if predicted and len(predicted) > 0 else 0)

    return m