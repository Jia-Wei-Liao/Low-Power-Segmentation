import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ============================== Helper function ==============================
def stats_graph(graph):
    flops = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print(f'FLOPs: {flops.total_float_ops}; Trainable params: {params.total_parameters}')

    return

# ============================== Set config ==============================
epochs          = 100
batch_size      = 64
num_workers     = 8
channel         = 8
depth           = 5
n_classes       = 6
lr              = 1e-3
lr_decay_period = 6
lr_decay_factor = 0.8
weight_decay    = 1e-4
resize_shape    = (192, 320)  # (H, W)

# ============================== Set dataLoader ==============================
class MTKSegDataset(Dataset):
    def __init__(self, input_path, label_path, label_name='', transform=None):
        super(MTKSegDataset, self).__init__()
        self.img_files = glob.glob(os.path.join(input_path, '*.jpg'))
        self.transforms = transform
        self.mask_files = list(map(
            lambda x: os.path.join(label_path, os.path.basename(x).replace('.jpg', f'{label_name}.png')),
            self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = cv2.imread(img_path)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = F.one_hot(torch.from_numpy(label).to(torch.int64), 6)

        if self.transforms:
            data = np.concatenate((image, label), axis=2)
            data = np.transpose(data, [2, 0, 1])
            data = self.transforms(torch.from_numpy(data).float())
            data = np.transpose(data, [1, 2, 0])
            image, label = data[:, :, 0:3], data[:, :, 3:9]

        return image, label


# ============================== Data transform ==============================
class RandomNoise():
    def __init__(self, sig=0.005, p=0.1):
        self.sig = sig
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            image = data[:3]
            data[:3] += self.sig * torch.randn(image.shape)

        return data


class RandomBrightness():
    def __init__(self, factor=0.1, p=0.1):
        self.p = p
        self.factor = factor

    def __call__(self, data):
        if random.random() <= self.p:
            factor = random.uniform(1-self.factor, 1+self.factor)
            image = data[:3]
            image = torchvision.transforms.functional.adjust_brightness(image, factor)
            data[:3] = torch.clip(image, min=0, max=1)

        return data


train_transforms = transforms.Compose([
    transforms.Resize(size=resize_shape),
    transforms.RandomHorizontalFlip(p=0.5),
    RandomNoise(sig=0.005, p=0.1),
    RandomBrightness(factor=0.1, p=0.1)
    ])

train_set = MTKSegDataset(
    'dataset/images', 'dataset/labels/class_labels', '_lane_line_label_id',
    transform=train_transforms)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_set = MTKSegDataset(
    'dataset/images_real_world', 'dataset/labels_real_world', '',
    transform=transforms.Resize(size=resize_shape))
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ============================== Construct model ==============================
input_img = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 3))
input_lab = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, n_classes))

x = tf.image.resize(input_img, resize_shape)
y = tf.image.resize(input_lab, resize_shape)

x = x / 255  # normalize

x = tf.layers.conv2d(x, channel, 3, 1, 'same')
x = tf.layers.batch_normalization(x, center=True, scale=True)
x = tf.nn.relu(x)
skip_connect = []

# Encoder
for d in range(depth):
    skip_connect.append(x)
    x = tf.layers.conv2d(x, channel*(2**(d+1)), 3, 1, 'same')
    x = tf.layers.batch_normalization(x, center=True, scale=True)
    x = tf.nn.relu(x)

    if d < depth-1:
        x = tf.layers.conv2d(x, channel*(2**(d+1)), 3, 2, 'same')

    else:
        x = tf.layers.conv2d(x, channel*(2**(d+1)), 3, 1, 'same')
        
    x = tf.layers.batch_normalization(x, center=True, scale=True)
    x = tf.nn.relu(x) 

# Decoder
for d in range(depth):
    if d > 0:
        x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.layers.conv2d(x, channel*(2**(depth-d-1)), 3, 1, 'same') + skip_connect[-d-1]
    x = tf.layers.batch_normalization(x, center=True, scale=True)
    x = tf.nn.relu(x)

out = tf.layers.conv2d(x, n_classes, 3, 1, 'same')

outputs = out
outputs = tf.image.resize(outputs, (1080, 1920))
outputs = tf.argmax(outputs, -1)

# ============================== Loss function ==============================
data_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y)
data_loss = tf.reduce_mean(data_loss)
reg_term =  tf.reduce_sum([tf.nn.l2_loss(w) for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
loss = data_loss + weight_decay * reg_term

# ============================== Optimization ==============================
tmp = int(len(train_set) / batch_size)
one_epoch_iter  = tmp if len(train_set) % batch_size == 0 else tmp + 1

global_step = tf.Variable(0, name='global_step', trainable=False)
lr_scheduler = tf.compat.v1.train.exponential_decay(
    lr, global_step,
    one_epoch_iter * lr_decay_period,
    lr_decay_factor, staircase=True)

train = tf.compat.v1.train.AdamOptimizer(
    learning_rate=lr_scheduler).minimize(loss, global_step=global_step)

pred = tf.argmax(out, -1)
lab = tf.argmax(y, -1)
acc, update_op = tf.metrics.mean_iou(lab, pred, n_classes)

saver = tf.compat.v1.train.Saver()

global_init = tf.compat.v1.global_variables_initializer()
local_init  = tf.compat.v1.local_variables_initializer()

sess = tf.Session()
sess.run(global_init)
sess.run(local_init)
stats_graph(tf.compat.v1.get_default_graph())

# ============================== Trainig step ==============================
def train_step(train_loader, sess):
    total_num  = 0
    total_loss = 0
    total_acc  = 0
    train_bar = tqdm(train_loader, desc=f'Training {ep}')
    for data in train_bar:
        images, labels = data
        images, labels = images.numpy(), labels.numpy()
        feed = {input_img: images, input_lab: labels}

        sess.run(local_init)
        get_loss, get_lr, _, _ = sess.run(
            [loss, lr_scheduler, train, update_op], feed_dict=feed)
        get_acc = sess.run(acc)

        total_num += batch_size
        total_loss += get_loss * batch_size
        total_acc += get_acc * batch_size
        train_bar.set_postfix({
            'lr'  : get_lr,
            'loss': total_loss / total_num,
            'acc' : total_acc / total_num
        })

    return


def val_step(val_loader, sess):
    total_num  = 0
    total_loss = 0
    total_acc  = 0
    val_bar = tqdm(val_loader, desc=f'Validation {ep}')
    for data in val_bar:
        images, labels = data
        images, labels = images.numpy(), labels.numpy()
        feed = {input_img: images, input_lab: labels}

        sess.run(local_init)
        get_loss, _ = sess.run([loss, update_op], feed_dict=feed)
        get_acc = sess.run(acc)

        total_num += batch_size
        total_loss += get_loss * batch_size
        total_acc += get_acc * batch_size
        val_bar.set_postfix({
            'loss': total_loss / total_num,
            'acc' : total_acc / total_num
        })

    return


for ep in range(1, epochs+1):
    train_step(train_loader, sess)
    val_step(val_loader, sess)
    saver.save(sess, 'checkpoint/')
    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_node_names=['ArgMax'])
    tf.io.write_graph(graph_def, '', f'checkpoint/ep={ep}_model_weight.pb', as_text=False)
