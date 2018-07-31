

from __future__ import print_function
from __future__ import division
import os
import cv2
import sys
import numpy as np
from tqdm import trange
import xml.etree.ElementTree as et

img_source = sys.argv[1]
xml_source = sys.argv[2]
dest = sys.argv[3]
images = [image for image in os.listdir(img_source) if image.endswith('.jpg')]
xmls = [xml for xml in os.listdir(xml_source) if xml.endswith('.txt')]
os.mkdir(dest)


for k in trange(len(images)):
    # name = '00001139.jpg'
    # file_name, ext = os.path.splitext(name)
    file_name, ext = os.path.splitext(images[k])
    root = et.parse(os.path.join(xml_source, file_name+'.txt')).getroot()
    img = cv2.imread(os.path.join(img_source, file_name+'.jpg'))

    tags = []
    bbox = []
    def recurse(root):
        for child in root:
            recurse(child)
        class_ = root.get('class')
        if class_ == 'ocr_par' or class_ == 'ocr_carea':
            bbox_line = root.get('title').split(';')[0].split(' ')[1:]
            bbox.append(map(int, bbox_line))
        if not class_ in tags:
            tags.append(class_)

    recurse(root)
    # print(tags)
    # print(bbox)
    save_path = os.path.join(dest, file_name)
    os.mkdir(save_path)
    for idx, line in enumerate(bbox):
        x1, y1, x2, y2 = line
        new = np.zeros_like(img)
        new[y1:y2,x1:x2,:] = img[y1:y2,x1:x2,:]
        # at this point, we have bounding boxes of entities
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_path, '{}_{}.png'.format(file_name, idx)), new)






















