import cv2
import os
import numpy as np

l = os.listdir('../predict/car_png')

for i, im_name in enumerate(l):

    car = cv2.imread('../predict/car_png/' + im_name)
    build = cv2.imread('../predict/train_stack_orig(with_road)+flip(3)_png/' + im_name)

    im_merge = np.zeros(shape=(1500, 1500, 3))
    im_merge[:, :, 0] = car[:, :, 0]
    im_merge[:, :, 1] = build[:, :, 1]

    cv2.imwrite('../predict/merge/' + im_name, im_merge)

    print(i, im_name)
