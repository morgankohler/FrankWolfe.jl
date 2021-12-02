import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from models import load_model, load_adfmodel
import instances


# GENERAL PARAMETERS
MODE = 'joint_untargeted'
IMG_SHAPE = [28, 28]

# LOAD MODEL
model = load_model()

generator = instances.load_generator()


def get_data_sample(index):
    return (
        generator[index],
        os.path.splitext(os.path.split(generator.filenames[index])[1])[0],
    )


def store_single_result(mapping, name, fname, rate):
    savedir = os.path.join('results', fname)
    os.makedirs(savedir, exist_ok=True)
    # print(mapping.shape)
    mapping = np.reshape(mapping, IMG_SHAPE)
    # for line in mapping:
    #     print(line)
    # raise Exception

    # for row in mapping:
    #     print(row)

    # np.save(f'/home/Morgan/fw-rde/mnist/results/{name}.npy', mapping)

    plt.imsave(
        os.path.join(
            savedir,
            f'{name}.png'
        ),
        mapping.squeeze(),
        cmap='Greys',
        vmin=np.min(mapping),
        vmax=np.max(mapping),
        format='png',
    )


def store_pert_img(x, s, p, name, fname, rate):
    savedir = os.path.join('results', fname)
    os.makedirs(savedir, exist_ok=True)
    # print(mapping.shape)
    x = np.reshape(x, IMG_SHAPE)
    s = np.reshape(s, IMG_SHAPE)
    p = np.reshape(p, IMG_SHAPE)

    x = x + s*p
    # for line in mapping:
    #     print(line)
    # raise Exception

    # np.save(f'/home/Morgan/fw-rde/mnist/results/{name}.npy', x)

    plt.imsave(
        os.path.join(
            savedir,
            f'{name}.png'
        ),
        x.squeeze(),
        cmap='Greys',
        vmin=np.min(x),
        vmax=np.max(x),
        format='png',
    )


def get_distortion(x, model=model, mode=MODE):

    x_tensor = tf.constant(x, dtype=tf.float32)
    s_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    s_tensor = tf.reshape(s_flat, x.shape)

    p_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    p_tensor = tf.reshape(p_flat, x.shape)

    pred = model.predict(x)
    node = np.argpartition(pred[0, ...], -2)[-1]
    # target = pred[0, node]

    network_input = x + s_tensor * p_tensor
    out = model(network_input)
    if mode == 'joint_untargeted':
        loss = tf.squeeze(out[..., node])

    gradient = K.gradients(loss, [s_flat, p_flat])
    f_out = K.function([s_flat, p_flat], [loss])
    f_gradient = K.function([s_flat, p_flat], [gradient])

    # a = tf.random.uniform(shape=s_flat.shape)
    # b = tf.random.uniform(shape=s_flat.shape)
    #
    # c = f_out([a, b])
    # d = f_gradient([a, b])

    return lambda s, p: f_out([s, p])[0], lambda s, p: f_gradient([s, p])[0][0], lambda s, p: f_gradient([s, p])[0][1], node, pred


# x, fname = get_data_sample(0)
#
# f, gs, gp, n, p = get_distortion(x)
#
# a = tf.random.uniform(shape=[28*28])
# b = tf.random.uniform(shape=[28*28])
#
# out = f(a,b)
#
#
# _=0