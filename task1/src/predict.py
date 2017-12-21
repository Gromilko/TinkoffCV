import tifffile as tiff
import numpy as np
from unet import get_unetBN
import os
from datetime import datetime

path = '../tif/tif_test_percentile'
weight = '../weights/building/12.12(1)_no_road/weights_ep103-vjc0.8456.hdf5'

img_list = os.listdir(path)

model = get_unetBN()
model.load_weights(weight)

start = datetime.now()

for count, IM_ID in enumerate(img_list[9:10]):

    im_rgb = tiff.imread(os.path.join(path, IM_ID))
    im_rgb = np.expand_dims(im_rgb, axis=0)
    im_size = im_rgb.shape[1:]
    im_size_orig = im_rgb.shape[1:]

    pad = 50

    # добавляем с краев к изображению куски, чтобы не было краевого эфекта
    # позже вспомнил о np.pad, но менять уже не стал.
    big_img = np.zeros(shape=(1, im_size[0]+256+pad, im_size[1]+256+pad), dtype=np.float32)
    big_img[:, pad:pad+im_size[0], pad:pad+im_size[1]] = im_rgb
    # левый верхний уголок
    big_img[:, :pad, :pad] = im_rgb[:, :pad, :pad]
    # верхняя полоса
    big_img[:, :pad, pad:pad+im_size[1]] = im_rgb[:, :pad, :]
    # левая полоса
    big_img[:, :pad, pad+im_size[1]:] = im_rgb[:, :pad, im_size[1] - 256:]
    # верхняя полоса
    big_img[:, pad:pad+im_size[0], :pad] = im_rgb[:, :, :pad]
    big_img[:, pad:pad+im_size[0], pad+im_size[1]:] = im_rgb[:, :, im_size[1] - 256:]
    big_img[:, pad+im_size[0]:, pad:pad+im_size[1]] = im_rgb[:, im_size[0]-256:, :]
    big_img[:, pad+im_size[0]:, :pad] = im_rgb[:, im_size[0]-256:, :pad]
    big_img[:, pad+im_size[0]:, pad+im_size[1]:] = im_rgb[:, im_size[0]-256:, im_size[1]-256:]

    # нормализация
    img = big_img - big_img.mean()
    img /= img.std()

    im = np.expand_dims(img, axis=0)
    im_size = img.shape[1:]

    full_mask_orig = np.zeros(shape=(im_size[0], im_size[1]), dtype=np.float32)

    print(str(count) + ' ' + IM_ID)

    for i, y in enumerate(range(0, im_size[0]-300, 256 - pad*2)):
        for j, x in enumerate(range(0, im_size[1]-300, 256 - pad*2)):
            # тут не оптимизированно мы передаем в predict по одному кропу за раз
            # правильней было бы нарезать всю картинку на кропы и передать одной пачкой
            output_orig = model.predict(im[:, :, x:x+256, y:y+256], batch_size=4, verbose=0)

            full_mask_orig[x+pad:x+256-pad, y+pad:y+256-pad] = output_orig[0, 0, pad:256-pad, pad:256-pad]


    mask_build_orig = full_mask_orig[50:50+im_size_orig[0], 50:50+im_size_orig[1]]

    mask_sum = mask_build_orig
    # mask_sum /= 4
    mask_sum = np.where(mask_sum > 0.98, 1.0, 0.0).astype(np.float32)
    tiff.imsave('../predict/train_no_road/{}'.format(IM_ID), mask_sum)

stop = datetime.now()
print("Время выполнения прогрммы: ")
print(stop - start)
