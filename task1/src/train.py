import numpy as np
import os
from unet import get_unetBN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from datetime import datetime

# 'car' or 'building'
CLASS_NAME = 'building'

if __name__ == "__main__":
    print('start load...')
    data = np.load('../data/build_train_data_flip_no_road.npy')
    label = np.load('../data/build_train_label_flip_no_road.npy')
    data_test = np.load('../data/build_test_data_flip_no_road.npy')
    label_test = np.load('../data/build_test_label_flip_no_road.npy')
    print('load complete')

    print('Get U-net...')
    model = get_unetBN()
    print('U-net start...')

    batch_size = 10
    epochs = 2

    date_str = datetime.now().strftime('%d.%m.%y')
    csv_logger = CSVLogger('../logs/{}_{}_train.csv'.format(CLASS_NAME, date_str), append=False)

    path = os.path.join('../weights', CLASS_NAME, date_str)
    # os.mkdir(path)
    model_checkpoint = ModelCheckpoint(os.path.join(path, "weights_ep{epoch:02d}-vjc{val_jaccard_coef:.4f}.hdf5"),
                                       monitor='val_jaccard_coef', verbose=1, save_best_only=True,
                                       save_weights_only=True, mode='auto', period=1)

    model.fit(data, label, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              callbacks=[csv_logger, model_checkpoint], validation_data=(data_test, label_test))
