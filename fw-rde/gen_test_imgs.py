import torchvision
import os
from PIL import Image
import tqdm

root = '/data/morgan'

ds = torchvision.datasets.STL10(root=root, split='test')

img_ctr = 0
for i in tqdm.tqdm(range(len(ds))):
    img = ds[i][0]
    label = ds[i][1]

    save_path = os.path.join(root, 'stl10', f'{label}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img.save(os.path.join(save_path, f'{img_ctr}.bmp'))

    img_ctr += 1
