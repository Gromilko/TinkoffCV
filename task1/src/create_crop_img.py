import tifffile as tiff
import numpy as np
import csv
import os
import sys

maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

data = []
label = []
data_test = []
label_test = []

# путь до изображений
PATH_IMG = '../tif/tif_train_percentile'
# путь до масок. Указать путь до масок машин или до масок зданий
PATH_MASK = '../masks/masks_build_train'
# не забыть поменять внизу путь до выходного файла

tif_list = os.listdir(PATH_IMG)

if __name__ == "__main__":
    count = 0
    for count, tif_name in enumerate(tif_list):
        img = tiff.imread(os.path.join(PATH_IMG, tif_name))
        mask = tiff.imread(os.path.join(PATH_MASK, tif_name))
        # .transpose([1, 2, 0])

        img_norm = img - img.mean()
        img_norm /= img_norm.std()

        im_size_x, im_size_y = img_norm.shape

        # с каждой картинки в рандомных местах вырезаем 50 кропов размером 256х256
        for i in range(50):
            # выбираем рандомно точку на изображении, чтобы от этой точки
            # вниз и в право можно было вырезать кроп разером 256х256
            x, y = np.random.randint(0, im_size_x-257), np.random.randint(0, im_size_y-257)
            im_crop = img_norm[x:x + 256, y:y + 256]
            ms_crop = mask[x:x + 256, y:y + 256]

            # 1/5 часть картинок оставляем для тестирования
            # на остальных будем обучать
            if np.random.randint(0, 10) == 5:
                data_test.append(im_crop)
                label_test.append(ms_crop)
            else:
                data.append(im_crop)
                label.append(ms_crop)

    label = np.array(label).reshape(int(len(label)), 1, 256, 256)
    data = np.array(data).reshape(int(len(data)), 1, 256, 256)
    np.save('../data/build_train_data.npy', data)
    np.save('../data/build_train_label.npy', label)
    print('train shape saved', data.shape, label.shape)

    label = np.array(label_test).reshape(int(len(label_test)), 1, 256, 256)
    data = np.array(data_test).reshape(int(len(data_test)), 1, 256, 256)
    np.save('../data/build_test_data.npy', data)
    np.save('../data/build_test_label.npy', label)
    print('test shape saved', data.shape, label.shape)

    print('Complete')
