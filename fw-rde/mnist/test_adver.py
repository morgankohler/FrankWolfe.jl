import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from models import load_model, load_adfmodel
from PIL import Image

# GENERAL PARAMETERS
MODE = 'diag'       # 'diag', 'half', or 'full'

# LOAD MODEL
print(os.getcwd())
adfmodel = load_adfmodel(mode=MODE)
model = load_model(path='mnist/mnist-convnet-avgpool-weights.hdf5')
#
# x = np.array(Image.open('mnist/results/untargeted_ktest/img.png'))
# x = (x - 37.96046)[:, :, 0]
#
# k = np.array(Image.open('mnist/results/untargeted_ktest/diag-mode-rate50-nx.png'))
# k = k[:, :, 0]

s = np.load('/home/Morgan/fw-rde/mnist/results/784.npy')
s = np.expand_dims(np.expand_dims(s, axis=0), axis=3)
x = np.load('/home/Morgan/fw-rde/mnist/results/x.npy')

print(np.max(x))
print(np.min(x))
noise=(1-s)
rand=np.random.normal(size=s.shape)
noise=noise*rand/np.max(rand)*np.max(x)
new = x + noise
new[new>np.max(x)] = np.max(x)
new[new<np.min(x)] = np.min(x)
# new = (new - np.min(new)) / (np.max(new) - np.min(new)) * (np.max(x) - np.min(x)) + np.min(x)
print(np.max(new))
print(np.min(new))
# new = np.expand_dims(new, axis=0)
# new = np.expand_dims(new, axis=3)

plt.figure()
plt.imshow(new.squeeze(), cmap='gray', vmin=np.min(new), vmax=np.max(new))
plt.show()
# new =

pred = model.predict(new)
print(pred)
_=0

