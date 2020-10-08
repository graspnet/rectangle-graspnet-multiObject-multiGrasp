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
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
from shapely.geometry import Polygon

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

def Rotate2D(pts,cnt,ang=scipy.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts-cnt, ar([[cos(ang), sin(ang)], [-sin(ang), cos(ang)]])) + cnt

def is_true_positive(angle, pred_polygon, theta, true_polygon_list, detected_gt_grasp_index_set):
    flag = False
    for i in range(theta.shape[0]):
        if abs(angle - theta[i]) >= 30.0:
            continue
        else:
            pred_area = pred_polygon.area
            true_area = true_polygon_list[i].area
            intersect_area = true_polygon_list[i].intersection(pred_polygon).area
            jaccard_index = intersect_area / (pred_area + true_area - intersect_area)
            if jaccard_index > 0.25:
                flag = True
                detected_gt_grasp_index_set.add(i)
    return flag

def count_true_positive(class_name, dets, theta, true_polygon_list, detected_gt_grasp_index_set, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0, 0

    angle_class = int(class_name[6:])
    angle = -90.0 + 10 * (angle_class - 1)
    total_num = 0
    true_positive_num = 0

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # plot rotated rectangles
        pts = ar([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        cnt = ar([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
        r_bbox = Rotate2D(pts, cnt, -pi/2-pi/18*(angle_class-1)) # rotated rectangle # maybe -pi/2-pi/18*(angle_class-1)
        pred_polygon = Polygon([(r_bbox[0,0],r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
        if is_true_positive(angle, pred_polygon, theta, true_polygon_list, detected_gt_grasp_index_set):
            true_positive_num += 1
        total_num += 1

    return true_positive_num, total_num

def get_true_grasps(scene_name, image_index):
    # Load the ground truth grasps
    scene_index = scene_name[-4:]
    if os.path.exists(os.path.join(graspnet_config.GRASPNET_ROOT, 'scenes', scene_name, graspnet_config.CAMERA_NAME, 'rectangle_grasp', scene_index)):
    # if os.path.exists('/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/' + scene_index):
        grasp_base_path = os.path.join(graspnet_config.GRASPNET_ROOT, 'scenes', scene_name, graspnet_config.CAMERA_NAME, 'rectangle_grasp', scene_index)
        # grasp_base_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/' + scene_index + '/'
    else:
        grasp_base_path = os.path.join(graspnet_config.GRASPNET_ROOT, 'scenes', scene_name, graspnet_config.CAMERA_NAME, 'rectangle_grasp')
        # grasp_base_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/'
    grasp_path = os.path.join(grasp_base_path, image_index + '.npy')
    # grasp_path = grasp_base_path + image_index + '.npy'
    grasp = np.load(grasp_path)
    mask = grasp[:, 5] <= 0.1
    grasp = grasp[mask]
    center_x = grasp[:, 0] # shape: (*, )
    center_y = grasp[:, 1] # shape: (*, )
    open_point_1_x = grasp[:, 2] # shape: (*, )
    open_point_1_y = grasp[:, 3] # shape: (*, )
    height = grasp[:, 4] # height of the rectangle, shape: (*, )
    theta = np.zeros(height.shape) # rotation angle of the rectangle. -90 ~ 90
    for i in range(theta.shape[0]):
        if center_x[i] > open_point_1_x[i]:
            theta[i] = np.arctan((open_point_1_y[i] - center_y[i]) / (center_x[i] - open_point_1_x[i])) * 180 / np.pi
        elif center_x[i] < open_point_1_x[i]:
            theta[i] = np.arctan((center_y[i] - open_point_1_y[i]) / (open_point_1_x[i] - center_x[i])) * 180 / np.pi
        else:
            theta[i] = -90.0
    alpha = theta + 90.0 # angle of the side of rectangle. 0 ~ 180
    open_point_2_x = 2 * center_x - open_point_1_x
    open_point_2_y = 2 * center_y - open_point_1_y

    vertex_1_x = open_point_1_x + height * np.cos(alpha) / 2
    vertex_1_y = open_point_1_y + height * np.sin(alpha) / 2
    vertex_2_x = 2 * open_point_1_x - vertex_1_x
    vertex_2_y = 2 * open_point_1_y - vertex_1_y
    vertex_3_x = open_point_2_x - height * np.cos(alpha) / 2
    vertex_3_y = open_point_2_y - height * np.sin(alpha) / 2
    vertex_4_x = 2 * open_point_2_x - vertex_3_x
    vertex_4_y = 2 * open_point_2_y - vertex_3_y

    true_polygon_list = []
    for i in range(theta.shape[0]):
        true_polygon = Polygon([(vertex_1_x[i], vertex_1_y[i]), (vertex_2_x[i], vertex_2_y[i]), (vertex_3_x[i], vertex_3_y[i]), (vertex_4_x[i], vertex_4_y[i])])
        true_polygon_list.append(true_polygon)
    return theta, true_polygon_list


def demo(sess, net, image_name, CONF_THRESHES):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', 'Images', image_name)
    im = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)

    scene_name = image_name[:10] # 'scene_0021'
    # scene_index = scene_name[-4:]
    image_index = image_name[11:15] # '0003'

    theta, true_polygon_list = get_true_grasps(scene_name, image_index)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)

    timer.toc()
    print('Detection took {:.3f}s'.format(timer.total_time))

    NMS_THRESH = 0.3
    num_conf_threshes = len(CONF_THRESHES)
    # Count the TP num and total num of the image
    image_true_positive_num = np.zeros(num_conf_threshes) # number of true positive proposed grasps in the image for EACH CONF_THRESH
    image_total_num = np.zeros(num_conf_threshes) # number of proposed grasps in the image for EACH CONF_THRESH
    image_total_gt_num = theta.shape[0] # the number of ground truth grasps in the image. IT IS A NUMBER, NOT ARRAY!
    detected_gt_grasp_index = [set() for i in range(num_conf_threshes)] # the list of sets of indexes of the detected ground truth grasps
    detected_gt_grasp_num = np.zeros(num_conf_threshes) # number of detected ground truth grasps in the image for EACH THRESH

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        class_true_positive_num = np.zeros(num_conf_threshes)
        class_total_num = np.zeros(num_conf_threshes)
        for i in range(num_conf_threshes):
            CONF_THRESH = CONF_THRESHES[i]
            class_true_positive_num[i], class_total_num[i] = count_true_positive(cls, dets, theta, true_polygon_list, detected_gt_grasp_index[i], thresh=CONF_THRESH)
        image_true_positive_num += class_true_positive_num
        image_total_num += class_total_num

    for j in range(num_conf_threshes):
        detected_gt_grasp_num[j] = len(detected_gt_grasp_index[j])

    return image_true_positive_num, image_total_num, image_total_gt_num, detected_gt_grasp_num


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


    CONF_THRESHES = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_conf_threshes = len(CONF_THRESHES)

    # im_names = []
    # for i in range(100, 190):
    #     for j in range(256):
    #         im_names.append('scene_{}+{}.png'.format(str(i).zfill(4), str(j).zfill(4)))

    im_names = os.listdir('..', 'data', 'demo', 'Images')
    im_names.sort()

    # im_names = ['scene_0108+0036.png','scene_0129+0058.png','scene_0156+0010.png','scene_0174+0092.png']
    num_images = len(im_names)

    total_true_positive_num = np.zeros(num_conf_threshes) # number of total TP grasps in the images
    total_proposed_num = np.zeros(num_conf_threshes) # number of all proposed grasps in the images
    total_gt_num = np.zeros(num_conf_threshes) # number of all ground truth grasps in the images
    total_detected_gt_grasp_num = np.zeros(num_conf_threshes) # number of all detected ground truth grasps in the images

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(im_name)
        image_true_positive_num, image_total_num, image_total_gt_num, detected_gt_grasp_num = demo(sess, net, im_name, CONF_THRESHES)
        total_true_positive_num += image_true_positive_num
        total_proposed_num += image_total_num
        total_gt_num += image_total_gt_num
        total_detected_gt_grasp_num += detected_gt_grasp_num

    total_false_positive_num = total_proposed_num - total_true_positive_num
    total_missed_gt_grasp_num = total_gt_num - total_detected_gt_grasp_num
    FPPI = total_false_positive_num / num_images
    miss_rate = total_missed_gt_grasp_num / total_gt_num

    print('========================================================================\n')
    print('Miss Rate: {}'.format(miss_rate))
    print('FPPI: {}'.format(FPPI))
    print('\n========================================================================')

    np.save('result/total_false_positive_num', total_false_positive_num)
    np.save('result/total_proposed_num', total_proposed_num)
    np.save('result/total_missed_gt_grasp_num', total_missed_gt_grasp_num)
    np.save('result/total_gt_num', total_gt_num)
    np.save('result/FPPI', FPPI)
    np.save('result/miss_rate', miss_rate)
