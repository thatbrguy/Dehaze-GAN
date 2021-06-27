import os
import cv2
import h5py
import numpy as np
from skimage.transform import resize

if __name__ == '__main__':

    if not os.path.exists('A'):
        os.mkdir('A')

    if not os.path.exists('B'):
        os.mkdir('B')

    with h5py.File('data.mat', 'r') as f:
        images = np.array(f['images'])
        depths = np.array(f['depths'])

    images = images.transpose(0, 1, 3, 2)
    depths = depths.transpose(2, 1, 0)
    depths = (depths - np.min(depths, axis = (0, 1))) / np.max(depths, axis = (0, 1))
    depths = ((1 - depths) * np.random.uniform(0.2, 0.4, size = (1449, ))).transpose(2, 0, 1)

    for i in range(len(images)):
        fog = (images[i] * depths[i]) + (1 - depths[i]) * np.ones_like(depths[i]) * 255
        fog = resize(fog.transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')
        img = resize(images[i].transpose(1, 2, 0), (256, 256, 3), mode = 'reflect')
        img = (img * 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fog = cv2.cvtColor(fog.astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join('A', str(i).zfill(4) + '.png'), fog)
        cv2.imwrite(os.path.join('B', str(i).zfill(4) + '.png'), img)
        
        print('Extracting image:', i, end = '\r')

    print('Done.')
