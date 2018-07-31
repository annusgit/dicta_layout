
"""
    Returns the dataloaders for dataset
"""

from __future__ import print_function
from __future__ import division
import os
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader


def get_dataloaders(base_path, batch_size):
    # Labels for our dataset
    # {charts:0, images:1, tables:2, maths:3, text:4}

    class dataset(Dataset):
        def __init__(self, data_dictionary):
            super(dataset, self).__init__()
            self.example_dictionary = data_dictionary
            print(len(self.example_dictionary))
            pass

        def __getitem__(self, k):
            # check 'k' index against the ranges of each path
            example_path, label = self.example_dictionary[k]
            example = torch.Tensor(cv2.imread(example_path).transpose(2,1,0))
            return {'input': example, 'label': label}

        def __len__(self):
            return len(self.example_dictionary)

    # train dirs with their labels
    train_original = 'original/new_gen_train'
    train_synthetic = 'synthetic/new_gen_train'
    test_original = 'original/new_gen_test'
    test_synthetic = 'synthetic/new_gen_test'
    train_data_paths = {
                        base_path + 'charts/' + train_original :  0,
                        base_path + 'charts/' + train_synthetic:  0,
                        base_path + 'images/' + train_original :  1,
                        base_path + 'tables/' + train_original :  2,
                        base_path + 'tables/' + train_synthetic:  2,
                        base_path + 'maths/'  + train_original :  3,
                        base_path + 'maths/'  + train_synthetic:  3,
                        base_path + 'text/'   + train_original :  4,
                        # we will use all synthetic dataset to train and validate on...
                        # base_path + 'charts/' + test_synthetic :  0,
                        # base_path + 'tables/' + test_synthetic:   2,
                        # base_path + 'maths/'  + test_synthetic:   3,
                        }
    # only original dataset will be used to report our test accuracy
    test_data_paths =  {
                        base_path + 'charts/' + test_original :  0,
                        base_path + 'images/' + test_original :  1,
                        base_path + 'tables/' + test_original :  2,
                        base_path + 'maths/'  + test_original :  3,
                        base_path + 'text/'   + test_original :  4
                        }

    # create training set examples to labels dictionary
    train_examples_dictionary = {}
    for path in train_data_paths.keys():
        content = os.listdir(path)
        for file in content:
            # for each index as key, we want to have its path and label as its...
            train_examples_dictionary[len(train_examples_dictionary)] = (os.path.join(path, file),
                                                                         train_data_paths[path])
    # at this point, we have our train data dictionary mapping file paths to labels
    # we can split it into two dicts if we want, for example
    keys = train_examples_dictionary.keys()
    random.shuffle(keys)
    train_dictionary, val_dictionary = {}, {}
    for l, key in enumerate(keys):
        if l % 15 == 0:
            val_dictionary[len(val_dictionary)] = train_examples_dictionary[key]
            continue
        train_dictionary[len(train_dictionary)] = train_examples_dictionary[key]

    # create testset examples to labels dictionary
    test_dictionary = {}
    for path in test_data_paths.keys():
        content = os.listdir(path)
        for file in content:
            # for each index as key, we want to have its path and label as its...
            test_dictionary[len(test_dictionary)] = (os.path.join(path, file), test_data_paths[path])
    # at this point, we have our test data dictionary mapping file paths to labels

    train_data = dataset(data_dictionary=train_dictionary)
    val_data = dataset(data_dictionary=val_dictionary)
    test_data = dataset(data_dictionary=test_dictionary)
    print('train examples =', len(train_dictionary), 'val examples =', len(val_dictionary),
          'test examples =', len(test_dictionary))

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)

    # print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    # for idx, data in enumerate(test_dataloader):
    #     examples, labels = data['input'], data['label']
    #     print('on batch {}/{}, {} -> {}'.format(idx+1, len(test_dataloader), examples.size(), labels))

    return train_dataloader, val_dataloader, test_dataloader


def main():
    get_dataloaders(base_path='/home/annus/Desktop/logical_layout_analysis/new_dataset/croped_dataset/',
                    batch_size=4)


if __name__ == '__main__':
    main()














