import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from models import cifar10vgg
import instances
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# GENERAL PARAMETERS
MODE = 'joint_untargeted'

IMG_SHAPE = [32,32,3]

# LOAD MODEL
model = cifar10vgg()
# model = load_model()
generator = instances.load_generator()


def get_data_sample(index):
    return (
        generator[index],
        os.path.splitext(os.path.split(generator.filenames[index])[1])[0],
    )


def store_single_result(mapping, var, fname, rate, d, test_name):
    # if not os.path.exists(savedir = os.path.join('results', test_name))
    #     os.mkdir(savedir = os.path.join('results', test_name))
    if rate == 0:
        savedir = os.path.join('results', test_name, fname)
    else:
        savedir = os.path.join('results', test_name, fname, f'{rate}')
    os.makedirs(savedir, exist_ok=True)
    # print(mapping.shape)the s
    mapping = np.reshape(mapping, IMG_SHAPE)
    mapping = mapping.squeeze()
    mapping = mapping[:,:,::-1]

    mapping = mapping - np.min(mapping)
    mapping = mapping / np.max(mapping)

    plt.imsave(
        os.path.join(
            savedir,
            f'{var}_rate-{rate}_d-{d}.png',
        ),
        mapping,
        # cmap='Greys',
        vmin=np.min(mapping),
        vmax=np.max(mapping),
        format='png',
    )


def store_s(s, fname, rate, d, test_name):
    if rate == 0:
        savedir = os.path.join('results', test_name, fname)
    else:
        savedir = os.path.join('results', test_name, fname, f'{rate}')
    os.makedirs(savedir, exist_ok=True)
    s = np.reshape(s, IMG_SHAPE)
    plt.imsave(
        os.path.join(
            savedir,
            f's_rate-{rate}_d-{d}.png',
        ),
        np.mean(s, axis=-1).squeeze(),
        cmap='Reds',
        vmin=0.0,
        vmax=1.0,
        format='png',
    )


def store_pert_img(x, s, p, fname, rate, d, test_name, optim):
    if rate == 0:
        savedir = os.path.join('results', test_name, fname)
    else:
        savedir = os.path.join('results', test_name, fname, f'{rate}')
    os.makedirs(savedir, exist_ok=True)
    # print(mapping.shape)
    x = np.reshape(x, IMG_SHAPE)
    s = np.reshape(s, IMG_SHAPE)
    p = np.reshape(p, IMG_SHAPE)

    if optim == 'joint':
        pert_x = x + s * p
    elif optim == 'univariate':
        pert_x = x + p
    else:
        raise Exception("optim not implemented")

    pert_x = pert_x.squeeze()
    pert_x = pert_x[:,:,::-1]

    pert_x = np.clip(pert_x, a_min=np.min(x), a_max=np.max(x))
    pert_x = pert_x - np.min(pert_x)
    pert_x = pert_x / np.max(pert_x)

    plt.imsave(
        os.path.join(
            savedir,
            f'pertimg_rate-{rate}_d-{d}.png'
        ),
        pert_x,
        # cmap='Greys',
        vmin=np.min(pert_x),
        vmax=np.max(pert_x),
        format='jpg',
    )


def store_saliency_importance(joint_s, rates, fname, d, test_name):
    savedir = os.path.join('results', test_name, fname)
    os.makedirs(savedir, exist_ok=True)
    joint_s = np.reshape(joint_s, [len(rates)] + IMG_SHAPE)
    joint_s = np.mean(np.sum(joint_s, axis=0), axis=-1)
    joint_s = (joint_s - np.min(joint_s)) / np.max(joint_s)

    plt.imsave(
        os.path.join(
            savedir,
            f'impmap_d-{d}.png'
        ),
        joint_s,
        cmap='Reds',
        vmin=np.min(joint_s),
        vmax=np.max(joint_s),
        format='png',
    )


def get_distortion(x, model=model, mode=MODE, optim="joint"):

    x_tensor = tf.constant(x, dtype=tf.float32)
    s_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    s_tensor = tf.reshape(s_flat, x.shape)
    # print(x.shape)

    p_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    p_tensor = tf.reshape(p_flat, x.shape)

    pred = model.predict(x)
    node = np.argpartition(pred[0, ...], -2)[-1]
    # target = pred[0, node]

    if optim == "joint":
        unprocessed = x + s_tensor * p_tensor
    elif optim == "univariate":
        unprocessed = x + p_tensor + s_tensor*0
    else:
        raise Exception("optim not implemented")

    network_input = tf.clip_by_value(unprocessed, clip_value_min=np.min(x), clip_value_max=np.max(x))
    out = model.model(network_input)
    if mode == 'untargeted':
        target_node = None
        loss = tf.squeeze(out[..., node])
    elif mode == 'targeted':
        class_li = list(range(10))
        class_li.remove(node)
        new_class = random.randint(0, 8)
        target_node = class_li[new_class]
        cel = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cel([target_node], out)
    else:
        raise Exception("mode not implemented")

    loss_all = tf.squeeze(out)
    f_out_all = K.function([s_flat, p_flat], [loss_all])

    gradient = K.gradients(loss, [s_flat, p_flat])
    f_out = K.function([s_flat, p_flat], [loss])
    f_gradient = K.function([s_flat, p_flat], [gradient])

    return lambda s, p: f_out([s, p])[0], lambda s, p: f_gradient([s, p])[0][0], lambda s, p: f_gradient([s, p])[0][1], node, target_node, lambda s, p: f_out_all([s, p])


def get_model_prediction(x, s, p, node, target_node, mode, optim, pred1):

    s = np.reshape(s, x.shape)
    p = np.reshape(p, x.shape)

    if optim == 'joint':
        norm = np.sum(np.abs(s*p))
        pert_input = x + s * p
    elif optim == 'univariate':
        norm = np.sum(np.abs(p))
        pert_input = x + p
    else:
        raise Exception("optim not implemented")

    pert_input = tf.convert_to_tensor(pert_input)
    pert_input = tf.clip_by_value(pert_input, clip_value_min=np.min(x), clip_value_max=np.max(x))

    pred0 = model.predict(x)
    # pred1 = model.predict(pert_input, steps=1)

    pred1 = np.asarray([i.tolist() for i in pred1])
            
    # print(type(pred0))
    # print(type(pred1))

    node0 = np.argpartition(pred0[0, ...], -2)[-1]
    node1 = np.argpartition(pred1[0, ...], -2)[-1]

    pred0_percent = tf.nn.softmax(pred0)[..., node0]
    pred1_old_class_percent = tf.nn.softmax(pred1)[..., node0]
    pred1_new_class_percent = tf.nn.softmax(pred1)[..., node1]

    # pred0_logit = pred0[..., node0]
    # pred1_logit = pred1[..., node0]
    labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    with tf.Session() as sess:
        print('\n------------------------\n')
        print(f'orig pred {labels[node0]}: {node0} | ',
              f'orig pred: {pred0_percent.eval()}% | ',
              f'pert target {labels[target_node]}: {target_node} | ',
              f'pert pred {labels[node1]}: {node1} | ',
              f'pert pred new class: {pred1_new_class_percent.eval()}% | ',
              f'pert pred old class: {pred1_old_class_percent.eval()}% | ',
              )
        print('\n------------------------\n')

    if mode == 'untargeted':
        return int(node0 != node1), norm
    elif mode == 'targeted':
        return int(target_node == node1), norm
    else:
        return 0, norm

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