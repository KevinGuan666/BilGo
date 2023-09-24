import os
import glob
import numpy as np
from PIL import Image

if __name__ == '__main__':
    train_files = glob.glob(os.path.join('vegetarian/train', '*', '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    result = []
    for file in train_files:
        image = Image.open(file).convert('RGB')
        image = np.array(image).astype(np.uint8)
        image = image/255.
        result.append(image)

    print(np.shape(result))
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)