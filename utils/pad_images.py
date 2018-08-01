

from __future__ import print_function
from __future__ import division

import sys
import cv2
import numpy as np


def main():
    image = cv2.imread(sys.argv[1])
    image = cv2.resize(image, (1024,1024))
    cv2.imshow('',image)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    main()





