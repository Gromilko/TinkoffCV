import tifffile as tif
import os
import numpy as np
a = tif.imread('../tif/tif_test_percentile/100044.tif')
b = tif.imread('D:/kaggle/TinkoffCV_orig_end_18.12/task1/png/tif_test_scale_percentile/100044.tif')

print((a == b).any())
'''
q = []
w = []
qwe = 0
for count, i in enumerate(l):
    a = tif.imread('../tif/tif_train/'+i)
    b = tif.imread('D:/kaggle/TinkoffCV_orig_end_18.12/task1/tif/tif_train/'+i)
    if (a==b).all():
        qwe += 1

print(qwe)
'''
