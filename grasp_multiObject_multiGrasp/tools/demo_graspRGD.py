#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon

import graspnet_config

pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array

CLASSES = ('__background__',
           'angle_01', 'angle_02', 'angle_03', 'angle_04', 'angle_05',
           'angle_06', 'angle_07', 'angle_08', 'angle_09', 'angle_10',
           'angle_11', 'angle_12', 'angle_13', 'angle_14', 'angle_15',
           'angle_16', 'angle_17', 'angle_18', 'angle_19')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_240000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}

def Rotate2D(pts,cnt,ang=scipy.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts-cnt,ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))+cnt

def vis_detections(ax, image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # plot rotated rectangles
        pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        angle = int(class_name[6:])
        r_bbox = Rotate2D(pts, cnt, -pi/2-pi/20*(angle-1)) # maybe -pi/2-pi/18*(angle-1)
        pred_label_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
        pred_x, pred_y = pred_label_polygon.exterior.xy

        plt.plot(pred_x[0:2],pred_y[0:2], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[1:3],pred_y[1:3], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[2:4],pred_y[2:4], color='k', alpha = 0.7, linewidth=1, solid_capstyle='round', zorder=2)
        plt.plot(pred_x[3:5],pred_y[3:5], color='r', alpha = 0.7, linewidth=3, solid_capstyle='round', zorder=2)

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
    im_rgb_file = os.path.join(graspnet_config.GRASPNET_ROOT, 'scenes', image_name[:10], graspnet_config.CAMERA_NAME, 'rgb', image_name[11:15] + '.png')
    # im_rgb_file = '/DATA2/Benchmark/graspnet/scenes/' + image_name[:10] + '/kinect/rgb/' + image_name[11:15] + '.png'
    im_rgb = cv2.imread(im_rgb_file, cv2.IMREAD_UNCHANGED)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)

    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    fig, ax = plt.subplots(figsize=(12, 12))
    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(ax, image_name, im_rgb, cls, dets, thresh=CONF_THRESH)

    plt.axis('off')
    plt.tight_layout()

    #save result
    savepath = os.path.join('..', 'data', 'demo', 'results_all_cls', str(image_name)[:-4])
    # savepath = '../data/demo/results_all_cls/' + str(image_name)[:-4]
    plt.savefig(savepath)

    plt.draw()

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

    im_names = ['scene_0108+0036.png','scene_0129+0058.png','scene_0156+0010.png','scene_0174+0092.png']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
