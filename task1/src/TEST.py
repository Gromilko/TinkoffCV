import os
try:
    os.mkdir('../masks/masks_build_train')
    print('Запись во вновь созданную папку ')
except OSError:
    print('Папка уже создана')