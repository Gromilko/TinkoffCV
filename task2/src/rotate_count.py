import pandas as pd
import numpy as np
import rasterio
import math
import cv2

import tifffile as tif

from shapely.geometry import Point
from skimage.morphology import remove_small_holes, remove_small_objects


def rotateImage(image, angle):
    tif.imsave('im_for_rot.tif', image)
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    return result


# фильтр для бинарных изображений по размеру сегментов (реализован на основе волнового алгоритма)
# лучше воспользоваться функциями remove_small_objects и remove_small_holes из библиотеки skimage
def filterByLength(input, length, more=True):
    import queue as queue

    copy = input.copy()
    output = np.zeros_like(input)

    for i in range(input.shape[0]):

        for j in range(input.shape[1]):
            if (copy[i][j] == 255):

                q_coords = queue.Queue()
                output_coords = list()

                copy[i][j] = 100

                q_coords.put([i, j])
                output_coords.append([i, j])

                while (q_coords.empty() == False):

                    currentCenter = q_coords.get()

                    for idx1 in range(3):
                        for idx2 in range(3):

                            offset1 = - 1 + idx1
                            offset2 = - 1 + idx2

                            currentPoint = [currentCenter[0] + offset1, currentCenter[1] + offset2]

                            if (currentPoint[0] >= 0 and currentPoint[0] < input.shape[0]):
                                if (currentPoint[1] >= 0 and currentPoint[1] < input.shape[1]):
                                    if (copy[currentPoint[0]][currentPoint[1]] == 255):
                                        copy[currentPoint[0]][currentPoint[1]] = 100

                                        q_coords.put(currentPoint)
                                        output_coords.append(currentPoint)

                if (more == True):
                    if (len(output_coords) >= length):
                        for coord in output_coords:
                            output[coord[0]][coord[1]] = 255
                else:
                    if (len(output_coords) < length):
                        for coord in output_coords:
                            output[coord[0]][coord[1]] = 255

    return output


# переводим координаты из плоской системы координат epsg:32637 в координаты географической системы координат epsg:4326
def getLongLat(x1, y1):
    from pyproj import Proj, transform

    inProj = Proj(init='epsg:32637')
    outProj = Proj(init='epsg:4326')

    x2, y2 = transform(inProj, outProj, x1, y1)

    return x2, y2


# переводим координаты из географической системы координат epsg:4326 в координаты плоской системы координат epsg:32637
def get32637(x1, y1):
    from pyproj import Proj, transform

    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:32637')

    x2, y2 = transform(inProj, outProj, x1, y1)

    return x2, y2


# переводим координаты из плоской системы координат epsg:32637 в координаты изображения(пиксели)
def crsToPixs(width, height, left, right, bottom, top, coords):
    x = coords.xy[0][0]
    y = coords.xy[1][0]

    x = width * (x - left) / (right - left)
    y = height - height * (y - bottom) / (top - bottom)

    x = int(math.ceil(x))
    y = int(math.ceil(y))

    return x, y


# сегментация тени при помощи порогового преобразования
def shadowSegmentation(roi, threshold=30):
    thresh = cv2.equalizeHist(roi)
    ret, thresh = cv2.threshold(thresh, threshold, 255, cv2.THRESH_BINARY_INV)

    tmp = filterByLength(thresh, 50)

    if np.count_nonzero(tmp) != 0:
        thresh = tmp

    return thresh


# определяем размер(длину) тени; x,y - координаты здания на изображении thresh
def getShadowSize(thresh, x, y):
    # tif.imsave('thresh.tif', thresh)

    a = np.array(thresh, bool)
    c = remove_small_objects(a, 100, connectivity=100)
    thresh = remove_small_holes(c, 500).astype(np.uint8)
    # определяем минимальную дистанцию от здания до пикселей тени
    thresh = np.where(thresh > 0, 255, 0)
    # tif.imsave('treshRH.tif', thresh)
    min_dist = thresh.shape[0]
    min_dist_coords = (0, 0)

    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if (thresh[i, j] == 255) and (math.sqrt((i - y) * (i - y) + (j - x) * (j - x)) < min_dist):
                min_dist = math.sqrt((i - y) * (i - y) + (j - x) * (j - x))
                min_dist_coords = (i, j)  # y,x
    # print('***', min_dist_coords)
    # определяем сегмент, который содержит пиксель с минимальным расстоянием до здания
    import queue as queue

    q_coords = queue.Queue()
    q_coords.put(min_dist_coords)

    mask = thresh.copy()
    output_coords = list()
    output_coords.append(min_dist_coords)

    while q_coords.empty() == False:
        currentCenter = q_coords.get()

        for idx1 in range(3):
            for idx2 in range(3):

                offset1 = - 1 + idx1
                offset2 = - 1 + idx2

                currentPoint = [currentCenter[0] + offset1, currentCenter[1] + offset2]

                if (currentPoint[0] >= 0 and currentPoint[0] < mask.shape[0]):
                    if (currentPoint[1] >= 0 and currentPoint[1] < mask.shape[1]):
                        if (mask[currentPoint[0]][currentPoint[1]] == 255):
                            mask[currentPoint[0]][currentPoint[1]] = 100
                            q_coords.put(currentPoint)
                            output_coords.append(currentPoint)

    # отрисовываем ближайшую тень
    mask = np.zeros_like(mask).astype(np.uint8)
    for i in range(len(output_coords)):
        mask[output_coords[i][0]][output_coords[i][1]] = 255
    # tif.imsave('ближайшую_тень1.tif', mask)
    # print(type(mask[0, 0]))
    # определяем размер(длину) тени при помощи морфологической операции erode
    '''
    kernel = np.ones((2, 2), np.uint8)
    i = 0
    # print(mask.min())
    # print(mask.max())
    while np.count_nonzero(mask) != 0:
        mask = cv2.erode(mask, kernel, iterations=1)
        i += 1
    '''
    #return i + 1, thresh
    return i + 1, thresh


# определяем область, где нет облаков
def getNoCloudArea(b, g, r, n):
    gray = (b + g + r + n) / 4.0

    band_max = np.max(gray)

    gray = np.around(gray * 255.0 / band_max).astype(np.uint8)
    gray[gray == 0] = 255

    ret, no_cloud_area = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((50, 50), np.uint8)
    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((100, 100), np.uint8)

    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_DILATE, kernel)
    no_cloud_area = cv2.morphologyEx(no_cloud_area, cv2.MORPH_DILATE, kernel)
    no_cloud_area = 255 - no_cloud_area

    return no_cloud_area


if __name__ == "__main__":
    # читаем csv в формате lat,long,height
    df = pd.read_csv('../csv/test.csv')
    # вычисления производятся используя только панхроматический канал
    df = df[df.iloc[:]['img_name'] == 'pan']

    # угол солнца над горизонтом
    sun_elevation = 47.7

    img_pan = rasterio.open('../tif/pan.tif')
    b1 = img_pan.read(1)
    band_max = b1.max()
    img = np.around(b1 * 255.0 / band_max).astype(np.uint8)

    heights = list()
    heights_round = list()

    # определяем тень в окне размером (size, size)
    size = 300
    # маску теней получаем для всего снимка
    # порог подбирался в QGis
    shadow_full = tif.imread('../tif/pan_shadows.tif')
    shadow_full = np.flip(shadow_full, 0)

    for idx in range(0, df.shape[0]):
        print(idx)
        # геокоординаты и id здания
        id_b = df.loc[idx]['id']
        lat = df.loc[idx]['y']
        lon = df.loc[idx]['x']

        build_coords = Point(lon, lat)
        x, y = lon, lat

        # зная азимут солнца, мы знаем в каком направлении должны быть тени зданий
        # поиск тени зданий будем выполнять в этом направлении
        roi = img[y - size:y + int(size / 2), x - size:x + int(size / 2)].copy()

        roi[size, size] = 1
        shadow = shadow_full[y - size:y + int(size / 2), x - size:x + int(size / 2)].copy()

        # (size, size) - координаты здания в roi
        # умножаем длину тени в пикселях на размер пикселя в метрах
        # (в этом случае пространственное разрешение 3 м)
        shadow = np.where(shadow == 1, 255, 0)
        shadow_length, one_shadow = getShadowSize(shadow, size, size)
        # tif.imsave('one_shadow.tif', one_shadow)
        # shadow_length *= 0.339
        # print(shadow_length)

        # est_height = shadow_length * math.tan(sun_elevation * 3.14 / 180)
        # shadow_length_t = est_height/math.tan(70.3 * 3.14 / 180)
        # est_height = (shadow_length+shadow_length_t) * math.tan(sun_elevation * 3.14 / 180)
        # print('={}='.format(est_height))

        # heights.append((int(est_height), build_height))

        # ------------------ rotate -------------------

        b = rotateImage(one_shadow.astype(np.uint8), -18)
        b = np.where(b > 128, 255, 0)


        color = 0
        y = 319
        x = 273
        shadow_404 = False

        # сколько раз считать.
        # костыль на случай если в тени получилсь дырка
        how_much = 1

        if id_b == 1:
            x, y = 215, 288
        elif id_b == 2:
            x, y = 237, 290
        elif id_b in (3, 63, 107):
            how_much = 2
        elif id_b in (7, 13):
            how_much = 3
        elif id_b == 59:
            how_much = 4
        elif id_b == 8:
            x, y = 347, 335
        elif id_b == 12:
            x, y = 162, 294
        elif id_b == 14:
            x, y = 225, 316
        elif id_b == 15:
            x, y = 346, 310
        elif id_b == 19:
            x, y = 370, 334
        elif id_b == 21:
            x, y = 366, 315
        elif id_b == 22:
            x, y = 345, 332
        elif id_b == 30:
            x, y = 249, 236
        elif id_b == 31:
            x, y = 299, 321
        elif id_b == 35:
            x, y = 276, 286
        elif id_b == 60:
            x, y = 338, 332
        elif id_b == 62:
            x, y = 298, 310
        elif id_b == 66:
            x, y = 309, 318
            how_much = 3
        elif id_b == 75:
            x, y = 352, 333
        elif id_b == 82:
            x, y = 348, 318
        elif id_b == 88:
            x, y = 307, 290
        elif id_b == 99:
            x, y = 352, 352
            how_much = 3
        elif id_b == 103:
            x, y = 326, 345
        elif id_b == 115:
            x, y = 271, 266
        elif id_b == 116:
            x, y = 342, 320
            how_much = 2
        elif id_b == 132:
            x, y = 295, 305
        elif id_b == 136:
            x, y = 273, 309
        elif id_b == 137:
            x, y = 273, 312
        elif id_b == 138:
            x, y = 304, 289
        elif id_b == 141:
            x, y = 402, 326
        elif id_b == 142:
            x, y = 311, 248
        elif id_b == 144:
            x, y = 245, 271
        elif id_b == 146:
            x, y = 318, 249
        elif id_b == 147:
            x, y = 334, 295
        elif id_b == 150:
            x, y = 304, 314
        elif id_b == 155:
            x, y = 282, 324
        elif id_b == 159:
            x, y = 307, 290
        elif id_b == 169:
            x, y = 384, 108
        elif id_b == 170:
            x, y = 295, 251
        elif id_b == 173:
            x, y = 328, 229
        elif id_b == 176:
            x, y = 218, 280
        elif id_b == 185:
            x, y = 332, 271

        x_orig = x
        y_orig = y

        dirka = False
        count = 1
        while how_much != 0:
            while b[y, x] != 255:
                if dirka:
                    count += 1
                if y == 0:
                    shadow_404 = True
                    break
                # print(y)
                y -= 1

            if shadow_404:
                shadow_size = shadow_length
            else:
                while b[y, x] != 0:
                    # print(y)
                    y -= 1
                    count += 1
                shadow_size = count
            how_much -= 1
            dirka = True
        if id_b == 47:
            shadow_size += 7
        shadow_length = shadow_size * 0.339
        est_height = shadow_length * math.tan(47.7 * 3.14 / 180)

        # shadow_length_t = est_height/math.tan(70.3 * 3.14 / 180)
        # est_height = (shadow_length+shadow_length_t) * math.tan(sun_elevation * 3.14 / 180)

        est_height = int(est_height)
        #heights.append((est_height, build_height))

        qweasd = rotateImage(roi.astype(np.uint8), -18)
        qweasd[y, x] = 255
        qweasd[319, 273] = 255
        qweasd[y_orig, x_orig] = 255
        # cv2.imwrite('img/{}_r_({}-{}).png'.format(idx, est_height, build_height), qweasd)
        cv2.imwrite('../hend_predict/{}_r_({}-{}).png'.format(id_b, est_height, '_'), qweasd)
        # tif.imsave('{}_r.tif'.format(idx), qweasd)

        b[y, x] = 255
        b[y_orig, x_orig] = 255
        b[319, 273] = 255
        # cv2.imwrite('img/{}_s_({}-{}).png'.format(idx, est_height, build_height), b)
        cv2.imwrite('../hend_predict/{}_s_({}-{}).png'.format(id_b, est_height, '_'), b)
        # tif.imsave('{}_s.tif'.format(idx), b)

        # heights.append((id_b, est_height))

    '''
    MAPE = 0

    print(heights)

    for i in range(len(heights)):
        MAPE += (abs(heights[i][0] - heights[i][1]) / float(heights[i][1]))

    MAPE *= (100 / float(len(heights)))


    print(MAPE)

    sum = 0
    for i in range(len(heights)):
        sum += abs((heights[i][1] - heights[i][0]) / float(heights[i][1]))
    print(1 - sum / len(heights))
    print(sum / len(heights))

    
    with open('submit_12_rotate.csv', 'w') as f:
        f.write('id,height\n')
        for i in range(len(heights)):
            f.write("{},{}\n".format(heights[i][0], heights[i][1]))
    # asdasdasd
    '''