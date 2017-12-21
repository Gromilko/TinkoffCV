import xml.etree.cElementTree as ET
import numpy as np
import tifffile as tiff
import os

full_mask_list = os.listdir('../masks/xml_train')

# рисовать квадраты 4х4 пикселя в центре машин
for count, name_xml in enumerate(full_mask_list):
    count_car = 0
    print(str(count), name_xml, end=' Count: ')
    xml_path = '../masks/xml_train/' + name_xml

    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xml_path, parser=xmlp)

    img = np.zeros(shape=(1500, 1500), dtype=np.float32)

    root = tree.getroot()
    img_full = np.zeros(shape=(1500, 1500, 3), dtype=np.uint8)
    for lmobj in root.findall('object'):
        deleted = lmobj.find('deleted').text
        if deleted == '1': continue

        occluded = lmobj.find('occluded').text

        nameobj = lmobj.find('name')
        if nameobj.text is None: continue

        name = nameobj.text
        if name == 'car':
            polygon = lmobj.find('polygon')
            pt = polygon.findall('pt')
            x1, x2, y1, y2 = [0]*4
            x, y = [], []

            for i in pt:
                x.append(int(i.find('x').text))
                y.append(int(i.find('y').text))
            x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

            count_car += 1

            x_m, y_m = int(x1 + (x2 - x1) / 2), int(y1 + (y2-y1)/2)
            qx, qy = 2, 2

            # на случай если машина окажется на краю снимка
            if x_m < 2:
                qx = x_m
            elif 1500 - x_m < 2:
                qx = 1500 - x_m
            if y_m < 2:
                qy = y_m
            elif 1500 - y_m < 2:
                qy = 1500 - y_m

            img[y_m-qy:y_m+qy, x_m-qx:x_m+qx] = np.ones(shape=(qy*2, qx*2))
    print(count_car)
    tiff.imsave('../masks/masks_car_train_point_4x4/{}.tif'.format(name_xml.split('.')[0]), img)