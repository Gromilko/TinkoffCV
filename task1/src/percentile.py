import os
from datetime import datetime
import tifffile as tif
import numpy as np

l = os.listdir('../tif/tif_test')
for count, i in enumerate(l):
    a = tif.imread('../tif/tif_test/'+i)
    tif.imsave('../tif/tif_test_percentile/'+i, np.clip(a, 159, 2009))

l = os.listdir('../tif/tif_train')
for count, i in enumerate(l):
    a = tif.imread('../tif/tif_train/'+i)
    tif.imsave('../tif/tif_train_percentile/'+i, np.clip(a, 159, 2009))