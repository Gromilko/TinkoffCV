from skimage.feature import blob_log
import tifffile as tiff
import os

full_mask_list = os.listdir('../predict/car_point_4x4')

with open('../predict/count_car.csv', 'w') as f:
    f.write('id,count\n')
    for count, name in enumerate(full_mask_list):
        print(count, name)
        image = tiff.imread('../predict/car_point_4x4/{}'.format(name))

        blobs_log = blob_log(image, min_sigma=2, max_sigma=3, num_sigma=2, threshold=0.01, overlap=0.95)
        # print(blobs_log.shape[0])
        f.write("{},{}\n".format(name.split('.')[0], blobs_log.shape[0]))
