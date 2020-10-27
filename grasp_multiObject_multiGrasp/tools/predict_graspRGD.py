#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy

import graspnet_config

pi = scipy.pi
dot = scipy.dot
sin = scipy.sin
cos = scipy.cos
ar = scipy.array

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Choose GPU

CLASSES = ('__background__',
           'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
           'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
           'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
           'angle_16', 'angle_17', 'angle_18', 'angle_19')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_240000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}

def Rotate2D(pts, cnt, ang=scipy.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts-cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])) + cnt

def add_predicted_grasps(class_name, dets, image_grasps, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0, 0

    angle_class = int(class_name[6:])

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # plot rotated rectangles
        height = bbox[3] - bbox[1]
        cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        open_point = ar([bbox[2], (bbox[1] + bbox[3])/2])
        rotated_open_point = Rotate2D(open_point, cnt, -pi/2-pi/18*(angle_class-1)) # rotated rectangle (eg. class2: -100 degree, 对应y向上坐标的100 degree or -80 degree)
        this_grasp = list(cnt) + list(rotated_open_point)
        this_grasp.append(height)
        this_grasp.append(score)
        this_grasp.append(-1)
        image_grasps.append(this_grasp)

def demo(sess, net, image_name, CONF_THRESH):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', 'Images', image_name)
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)

    scene_name = image_name[:10] # 'scene_0021'
    image_index = image_name[11:15] # '0003'

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)

    timer.toc()

    NMS_THRESH = 0.3

    image_grasps = []

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        add_predicted_grasps(cls, dets, image_grasps, thresh=CONF_THRESH)

    image_grasps = np.array(image_grasps)
    np.save(os.path.join('..', 'predicted_rectangle_grasp', scene_name, graspnet_config.CAMERA_NAME, image_index), image_grasps)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('..', 'output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 20,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))


    CONF_THRESH = 0.0

    for scene_index in range(100, 190):
        scene_name = 'scene_{}'.format(str(scene_index).zfill(4)) # 'scene_0012'
        if not os.path.exists(os.path.join('..', 'predicted_rectangle_grasp', scene_name)):
            os.mkdir(os.path.join('..', 'predicted_rectangle_grasp', scene_name))
        if not os.path.exists(os.path.join('..', 'predicted_rectangle_grasp', scene_name, graspnet_config.CAMERA_NAME)):
            os.mkdir(os.path.join('..', 'predicted_rectangle_grasp', scene_name, graspnet_config.CAMERA_NAME))
        print(scene_name)
        for img_index in range(256):
            im_name = '{}+{}.png'.format(scene_name, str(img_index).zfill(4))
            demo(sess, net, im_name, CONF_THRESH)
