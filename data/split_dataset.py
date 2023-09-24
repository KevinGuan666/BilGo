import os
import glob
import random
import shutil
from PIL import Image


if __name__ == '__main__':
    test_split_ratio = 0.05
    target_size = 128
    raw_train_path = './vegetarian/raw_train'

    dirs = glob.glob(os.path.join(raw_train_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print(f'Totally {len(dirs)} classes: {dirs}')

    for path in dirs:
        path = path.split('\\')[-1]

        os.makedirs(f'vegetarian/train/{path}', exist_ok=True)
        os.makedirs(f'vegetarian/test/{path}', exist_ok=True)

        files = glob.glob(os.path.join(raw_train_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_train_path, path, '*.JPG'))
        files += glob.glob(os.path.join(raw_train_path, path, '*.jpeg'))
        files += glob.glob(os.path.join(raw_train_path, path, '*.png'))

        random.shuffle(files)

        boundary = int(len(files) * test_split_ratio)

        for i, file in enumerate(files):
            image = Image.open(file).convert('RGB')

            old_size = image.size

            ratio = float(target_size)/max(old_size)

            new_size = tuple([int(x*ratio) for x in old_size])

            img = image.resize(new_size, Image.ANTIALIAS)

            new_img = Image.new("RGB", (target_size, target_size))

            new_img.paste(img, ((target_size - new_size[0]) // 2,
                                    (target_size - new_size[1]) // 2))

            assert new_img.mode == 'RGB'

            if i <= boundary:
                new_img.save(os.path.join(f'vegetarian/test/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_img.save(os.path.join(f'vegetarian/train/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))

    test_files = glob.glob(os.path.join('vegetarian/test', '*', '*.jpg'))
    train_files = glob.glob(os.path.join('vegetarian/train', '*', '*.jpg'))

    print(f'Totally {len(test_files)} files for test')
    print(f'Totally {len(train_files)} files for train')



