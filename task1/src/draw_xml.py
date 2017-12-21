import xml.etree.cElementTree as ET
import numpy as np
import cv2
import tifffile as tiff
import os


def drawPolygon(etelem, img, name):
    pts = []
    for pt in etelem.findall('pt'):

        num = pt.find('x').text
        tx = int(float(num))

        num = pt.find('y').text
        ty = int(float(num))

        pts.append((tx, ty))

    pts = np.asarray(pts)
    pts = pts.reshape((-1, 1, 2))
    
    # buildings
    if 'b' in name:
        img = cv2.fillPoly(img, [pts], (0, 255, 0))
    # cars
    elif 'c' in name:
        img = cv2.fillPoly(img, [pts], (255, 0, 0))

    return img

if __name__ == "__main__":

    DIR_MASKS = '../masks/xml_train'
    # 'car' or 'building'
    TYPE = 'car'
    full_mask_list = os.listdir(DIR_MASKS)

    for i, name_xml in enumerate(full_mask_list):
        print(str(i), ' ', name_xml)
        xml_path = os.path.join(DIR_MASKS, name_xml)

        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_path, parser=xmlp)

        img = np.zeros(shape=(1500, 1500, 3), dtype=np.uint8)

        root = tree.getroot()
        img_full = np.zeros(shape=(1500, 1500, 3), dtype=np.uint8)
        count = 0
        for lmobj in root.findall('object'):
            deleted = lmobj.find('deleted').text
            if deleted == '1': continue

            occluded = lmobj.find('occluded').text

            nameobj = lmobj.find('name')
            if nameobj.text is None: continue

            name = nameobj.text
            if name == TYPE:
                # две следующие строки для того, чтобы на первой обучающей маске небыло дороги
                if TYPE == 'building' and name_xml == '100001.xml' and count == 81:
                    print('*')
                    continue
                polygon = lmobj.find('polygon')
                img_full += drawPolygon(polygon, img, name)
                count += 1

        # TODO: проверка на существование папки
        if TYPE == 'building':
            save_img = np.where(img_full[:, :, 1] < 1, 0, 1)
            tiff.imsave('../masks/masks_build_train/{}.tif'.format(name_xml.split('.')[0]), save_img)
        elif TYPE == 'car':
            save_img = np.where(img_full[:, :, 0] < 1, 0, 1)
            tiff.imsave('../masks/masks_car_train/{}.tif'.format(name_xml.split('.')[0]), save_img)
