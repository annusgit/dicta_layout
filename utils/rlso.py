

from __future__ import print_function
from __future__ import division

from tqdm import trange
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as pl
import PIL.Image as Image
import skimage.measure


def invert(image):
    inverted = np.zeros_like(image)
    inverted[image == 0] = 255
    return inverted


def binarize_image(image):
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return binary


def rlso(image, threshold, direction):
    if direction == 'right':
        i = 0
        while i < image.shape[0]:
            j = 0
            while j < image.shape[1]:
                if image[i,j] == 0:
                    adjacent = list(image[i,j+1:j+1+threshold])
                    if 0 in adjacent:
                        image[i,j+1:j+1+threshold] = 0
                        j += threshold
                    pass
                j += 1
            i += 1
    elif direction == 'left':
        i = image.shape[0]-1
        while i >= 0:
            j = image.shape[1]-1
            while j >= 0:
                if image[i,j] == 0:
                    adjacent = list(image[i,j-1-threshold:j-1])
                    if 0 in adjacent:
                        image[i,j-1-threshold:j-1] = 0
                        j -= threshold
                    pass
                j -= 1
            i -= 1
    if direction == 'down':
        i = 0
        while i < image.shape[1]:
            j = 0
            while j < image.shape[0]:
                if image[i,j] == 0:
                    adjacent = list(image[j+1:j+1+threshold,i])
                    if 0 in adjacent:
                        image[j+1:j+1+threshold,i] = 0
                        j += threshold
                    pass
                j += 1
            i += 1
    elif direction == 'up':
        i = image.shape[1]-1
        while i >= 0:
            j = image.shape[0]-1
            while j >= 0:
                if image[i,j] == 0:
                    adjacent = list(image[j-1-threshold:j-1,i])
                    if 0 in adjacent:
                        image[j-1-threshold:j-1,i] = 0
                        j -= threshold
                    pass
                j -= 1
            i -= 1

    return image


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # print(overlap)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def get_rect(img, contours, thresh):
    boxes = np.zeros(shape=(len(contours),4))
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (x,y) <= (10,10) and (w,h) >= (1024-10,1024-10):
            continue
        boxes[idx,0] = x
        boxes[idx,1] = y
        boxes[idx,2] = x+w
        boxes[idx,3] = y+h
    boxes = non_max_suppression_fast(boxes, overlapThresh=thresh)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    return img


def majority(array):
    (values, counts) = np.unique(array, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source')
    parser.add_argument('--dest', dest='dest')
    parser.add_argument('--thresh', dest='thr')
    args = parser.parse_args()
    source = args.source
    dest = args.dest
    thr = int(args.thr)

    os.mkdir(dest)
    bin_path, erod_path, and_path, rect_path = [os.path.join(dest, x)
                                                for x in ['binary', 'eroded', 'anded', 'rected']]
    map(os.mkdir, [bin_path, erod_path, and_path, rect_path])
    print(bin_path, erod_path, and_path, rect_path)

    file_list = [x for x in os.listdir(source) if x.endswith('.jpg')]
    for i in trange(len(file_list)):
        file_name = os.path.join(source, file_list[i])
        image = cv2.imread(file_name)
        image = cv2.resize(image, (1024, 1024))

        gray = Image.fromarray(image).convert('L')
        binary = np.asarray(gray).copy()

        bin_thresh = binary.max()-binary.mean()/2
        binary[binary < bin_thresh] = 0
        binary[binary != 0] = 255

        kernel = np.ones((5,5),np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=2)

        right = rlso(eroded, thr, direction='right')
        left = rlso(eroded, thr, direction='left')
        down = rlso(eroded, thr, direction='down')
        up = rlso(eroded, thr, direction='up')
        anded1 = np.bitwise_and(right, left)
        anded2 = np.bitwise_and(up, down)
        anded = np.bitwise_and(anded1, anded2)

        # this is important
        anded[0, :] = 255
        anded[:, 0] = 255
        anded[anded.shape[0] - 1, :] = 255
        anded[:, anded.shape[1] - 1] = 255

        # connect them now
        _, contours, hierarchy = cv2.findContours(anded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rected = get_rect(image.copy(), contours, thresh=0.5)

        cv2.imwrite(os.path.join(bin_path, file_list[i]), binary)
        cv2.imwrite(os.path.join(erod_path, file_list[i]), eroded)
        cv2.imwrite(os.path.join(and_path, file_list[i]), anded)
        cv2.imwrite(os.path.join(rect_path, file_list[i]), rected)
    pass


if __name__ == '__main__':
    main()





















