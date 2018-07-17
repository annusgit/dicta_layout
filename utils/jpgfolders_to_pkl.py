

from __future__ import print_function
from __future__ import division

import PIL.Image as Image
import numpy as np
import argparse
import pickle
import os


def get(data_path, label_path):
    examples, labels = [], []
    for x in os.listdir(data_path):
        ex_path = os.path.join(data_path, x)
        lab_path = os.path.join(label_path, x)
        ex = np.asarray(Image.open(ex_path))
        label = np.asarray(Image.open(lab_path))
        examples.append(ex)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', dest='example_path')
    parser.add_argument('--labels', dest='labels_path')
    args = parser.parse_args()
    example_path = args.example_path
    labels_path = args.labels_path
    examples, labels = get(data_path=example_path, label_path=labels_path)
    dataset = {'examples': examples, 'labels': labels}
    with open('../1_100.pkl', 'wb') as pkl:
        pickle.dump(dataset, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print(examples.shape, labels.shape)

if __name__ == '__main__':
    main()












