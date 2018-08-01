

"""
    Parse the coordinates from xml files and generate the dataset
"""

from __future__ import print_function
from __future__ import division
import os
import cv2
import numpy as np
import scipy.misc
import pickle as pkl
from tqdm import trange
import matplotlib.pyplot as pl
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as et


base = '/home/annus/Desktop/logical_layout_analysis/prima/'
source = os.path.join(base, 'fullPrimadataset')
dest = os.path.join(base, 'resized_images_and_all_labels')
examples, labels_path, check = os.path.join(dest, 'examples'), os.path.join(dest, 'labels'), \
                               os.path.join(dest, 'check')
xmls = [x for x in os.listdir(source) if x.endswith('.xml')]
print('total xmls = {}'.format(len(xmls)))
all_unique_labels = []

def make_dir(path):
    ##3 :)
    try:
        os.mkdir(path)
    except:
        pass

# make the destination folders
map(make_dir, (dest, examples, labels_path))

# with open(os.path.join(base, 'labels_list.pkl'), 'rb') as this:
#     all_labels = pkl.load(this)
def label_to_idx(iter):
    idx = {}
    for this_label in iter:
        idx[this_label] = len(idx)+1
    return idx

# self define it for better something
# all_labels = ['TableRegion', 'ImageRegion', 'ChartRegion', 'TextRegion', 'MathsRegion']
# all_labels = label_to_idx(all_labels)
# print(all_labels)
# all_labels = {'TableRegion':1, 'ImageRegion':2, 'GraphicRegion':2, 'ChartRegion':3, 'TextRegion':4, 'MathsRegion':5}
# print(all_labels)
all_labels = {}
# we need these to save our dataset
data_images = []
data_labels = []

def generate_label_map(polygons, dim):
    # polygon = [(100, 100), (200, 250), (300, 300)]  # or [x1,y1,x2,y2,...]
    img = np.zeros((dim, dim))
    # img = Image.new('L', (width, height), 0)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for label_name, polygon in polygons:
        # if we already have the id, let it be, if not, insert a new one!!!
        if label_name not in all_labels.keys():
            all_labels[label_name] = len(all_labels)+1 # we are leaving zero for background or something like that...
        id = all_labels[label_name]
        if len(polygon) < 2:
            print('missing this one!')
            continue
        draw.polygon(polygon, fill=id)
            # print(points[0])
            # draw.text((x, y),"Sample Text",(r,g,b))
            # if you want to write text
            # font = ImageFont.truetype("FreeSans.ttf", 14)
            # draw.text(polygon[0], label_name, 255, font=font)
        # else:
        #     pass
            # all_labels[label_name] = len(all_labels)+1
            # id = all_labels[label_name]
            # print('found new label: {} = {}'.format(label_name, all_labels[label_name]))
        # if 'id' in locals():
            # draw.polygon(polygon, fill=id)
            # font = ImageFont.truetype("FreeSans.ttf", 14)
            # # print(points[0])
            # # draw.text((x, y),"Sample Text",(r,g,b))
            # draw.text(polygon[0], label_name, 255, font=font)

    mask = np.array(img)
    # pl.imshow(mask)
    # pl.show()
    return mask


def pad_image(img, new_dim, channels):
    h, w, c = img.shape
    new = np.zeros(shape=(new_dim, new_dim, channels))
    w1 = new_dim // 2 - w // 2
    new[w1:new_dim-w1,w1:new_dim-w1,:] = img
    return new


def process_one(name):
    """
        Needs a base name of the file to process and draw the labels on the image
    :param name: name of the file
    :return: void, but saves the labelled image in dest folder
    """
    img = Image.open(os.path.join(source, name+'.jpg')).convert('RGB') # .convert('LA') #
    # we need this for resizing
    w, h = img.size
    new_dim = 900
    img = img.resize((new_dim, new_dim), Image.ANTIALIAS)
    img_copy = np.asarray(img.copy()) # [:,:,0] # .convert('1', dither=Image.NONE) to convert to binary image
    # save our example
    draw = ImageDraw.Draw(img)
    root = et.parse(os.path.join(source, 'pc-'+name+'.xml')).getroot()
    # print(root)
    page = root[-1]
    # print(name)
    # labels = []
    all_polygons_in_region = []
    label_tags = []
    for region in page:
        points = []
        for coords in region:
            region_label = region.tag.split('}')[-1]
            if region_label == 'TextRegion':
                # go deeper if it's text
                if 'type' in region.attrib:
                    region_label = region.attrib['type']
                # print(region_label)
            label_tags.append(region_label)
            # labels.append(region_label)
            for point in coords:
                # resize and append
                points.append((int(new_dim/w*int(point.get('x'))), int(new_dim/h*int(point.get('y')))))
                # print(region, coords, point)
                pass

        all_polygons_in_region.append(points)
        # draw.polygon((points), outline=1)

    # img.save(os.path.join(examples, '{}.jpg'.format(name)))
    mask = generate_label_map(polygons=zip(label_tags, all_polygons_in_region), dim=new_dim)
    if mask is None:
        print('missing this one: {}'.format(name + '.jpg'))
        return

    # we have our image and labels, let's pad them before saving them
    mask = np.expand_dims(mask, axis=2)
    mask = pad_image(mask, 1200, 1)[:,:,0]
    img_copy = pad_image(img_copy, 1200, 3)

    cv2.imwrite(os.path.join(examples, '{}.png'.format(name)), img_copy) # save #1. image as npy array
    cv2.imwrite(os.path.join(labels_path, '{}.png'.format(name)), mask) # save #2. label
    # np.save(os.path.join(labels_path, name), mask) # save #2. label
    # save them as np arrays instead
    # data_images.append(np.asarray(img_copy))
    # data_labels.append(mask)
    # draw and save in check
    # baad mein...


def check_label_generated(name):

    """
        Checks the labels generated earlier
    :return: None, but saves a labeled image in "check" folder for evaluation of labelling process
    """
    img = Image.open(os.path.join(examples, name+'.jpg')) # .convert('1', dither=Image.NONE) for single channel binary!
    label_image = Image.open(os.path.join(labels_path, name+'.jpg'))
    label_npy = np.asarray(label_image)
    label_img = label_image.convert('RGBA')
    print(img.size, label_img.size)
    pl.subplot(121)
    pl.imshow(img)
    pl.subplot(122)
    pl.imshow(label_npy)
    pl.show()


# check for one example
# process_one(name='00000194')
# check_label_generated('00000088')


if __name__ == '__main__':
    file_list = [x for x in os.listdir(source) if x.endswith('.jpg')]
    for i in trange(len(file_list)):
        name, extention = os.path.splitext(file_list[i])
        process_one(name=name)
    # # and finally pickle it up!
    # print('log: pickling now...')
    # dataman = {'examples': np.asarray(data_images), 'labels': np.asarray(data_labels)}
    # with open(os.path.join(base, '1_100_PRImA_dataset.pkl'), 'wb') as dataset:
    #     pkl.dump(dataman, dataset, protocol=pkl.HIGHEST_PROTOCOL)
    #     print(dataman['examples'].shape, dataman['labels'].shape)
    with open(os.path.join(base, 'labels_list.pkl'), 'wb') as this:
        pkl.dump(all_labels, this, pkl.HIGHEST_PROTOCOL)












