

from __future__ import print_function
from __future__ import division

import PIL.Image as Image
import pytesseract
import sys
import cv2


file_name = sys.argv[1]
# text = pytesseract.image_to_string(Image.open(file_name))
ret = pytesseract.image_to_boxes(Image.open(file_name))

with open('result.txt', 'w') as file:
    file.write(ret.encode('utf-8', 'ignore'))

with open('result.txt', mode='r') as new_file:
    read = new_file.readlines()

# remove those \n at the end
read = [x[0:-1] for x in read if x != '\n']
read = reversed(read)
img = cv2.imread(file_name)
img = cv2.flip(img, 0)

for line in read:
    line = line.split()
    x1, y1, x2, y2 = map(int, [str(line[1]), str(line[2]), str(line[3]), str(line[4])])
    # print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
img = cv2.flip(img, 0)
cv2.imwrite('result.png', img)




