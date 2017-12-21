import tifffile as tiff
import numpy as np
import os
import cv2

# 'car' or 'building'
TYPE = 'car'
band = 0 if TYPE == 'car' else 1

im_list = os.listdir('../predict/car')

for i in im_list:
    png = np.zeros(shape=(1500, 1500, 3), dtype=np.uint8)
    tif_im = tiff.imread('../predict/car/{}'.format(i))
    # tif_im = np.where(tif_im > 0.98, 1.0, 0.0).astype(np.float32)

    png[:, :, band] = np.where(tif_im == 1, 255, 0)
    cv2.imwrite('../predict/car_png/{}.png'.format(i.split('.')[0]), png)
