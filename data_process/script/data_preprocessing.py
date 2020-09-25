import numpy as np
import os
import cv2
import random

# scene_name_list = os.listdir('../scenes/') # ['scene_0023', 'scene_0022', 'scene_0021', ...]
# scene_name_list.sort()
scene_name_list = []
#for i in range(100):
for i in range(100, 190):
    scene_name_list.append('scene_{}'.format(str(i).zfill(4)))

print('************************************************')
print('************************************************')
print('The number of scenes is: {}'.format(len(scene_name_list)))
print('************************************************')
print('************************************************')

train_total_number = 0 # number of images used for training in all scenes
test_total_number = 0 # number of images used for testing in all scenes

f_scene_skip = open('scene_skip.txt', 'a')
f_image_skip = open('image_skip.txt', 'a')

for scene_name in scene_name_list:
    scene_index = scene_name[-4:]
    # if int(scene_index) < 100:
    #     is_training = True
    # else:
    #     is_training = False

    print('------------------------------------------')
    print('Processing scene: {}'.format(scene_index))
    try:
        rgb_name_list = os.listdir('/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rgb/') # ['0001.png', '0002.png', ...]
        rgb_name_list.sort()
    except:
        print('scene {} skipped'.format(scene_index))
        f_scene_skip.write('{}\n'.format(scene_name))
        continue

    # list used to record
    log_train_list = []
    log_test_list = []

    grasp_num = 256 # number of images with rectangle grasp in the scene
    test_num = int(grasp_num / 5) # number of images used for testing in the scene
    train_num = grasp_num - test_num # number of images used for training in the scene
    test_count = 0
    train_count = 0

    if os.path.exists('/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/' + scene_index):
        grasp_base_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/' + scene_index + '/'
    else:
        grasp_base_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rectangle_grasp/'

    for rgb_name in rgb_name_list:
        img_index = rgb_name[:4] # 0000 / 0001 / 0002 ...
        rgb_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/rgb/' + rgb_name # '../scenes/scene_0021/kinect/rgb/0000.png'
        depth_path = '/DATA2/Benchmark/graspnet/scenes/' + scene_name + '/kinect/depth/' + rgb_name # '../scenes/scene_0021/kinect/depth/0000.png'
        grasp_path = grasp_base_path + img_index + '.npy' # '../scenes/scene_0021/kinect/rectangle_grasp/(0021/)0000.npy'
        img_name = scene_name + '+' + img_index # 'scene_0021+0000'
        img_path = '../grasp_data/Images/' + img_name + '.png' # '../grasp_data/Images/scene_0021+0000.png'
        anno_path = '../grasp_data/Annotations/' + img_name + '.txt' # '../grasp_data/Annotations/scene_0021+0000.txt'

        try:
            grasp = np.load(grasp_path) # load the grasp annotationos, shape: (*, 6)
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED) # load the RGB information
            b_img, g_img, r_img = cv2.split(rgb_img)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # load the depth information
        except:
            print('{} skipped'.format(img_name))
            f_image_skip.write('{}\n'.format(img_name))
            continue

        reshaped_r_img = r_img.reshape((r_img.shape[0], r_img.shape[1], 1))
        reshaped_g_img = g_img.reshape((g_img.shape[0], g_img.shape[1], 1))
        reshaped_depth_img = depth_img.reshape((depth_img.shape[0], depth_img.shape[1], 1))
        # img = np.concatenate((rgb_img, reshaped_depth_img), axis=-1) # RGB-D
        img = np.concatenate((reshaped_depth_img, reshaped_g_img, reshaped_r_img), axis=-1) # RGD

        # Processing grasp
        mask = grasp[:, 5] <= 0.1
        grasp = grasp[mask]
        center_x = grasp[:, 0] # shape: (*, )
        center_y = grasp[:, 1] # shape: (*, )
        open_point_x = grasp[:, 2] # shape: (*, )
        open_point_y = grasp[:, 3] # shape: (*, )
        height = grasp[:, 4] # height of the rectangle, shape: (*, )
        score = grasp[:, 5] # shape: (*, )


        width = 2 * np.sqrt(np.square(open_point_x - center_x) + np.square(open_point_y - center_y)) # width of the rectangle, shape: (*, )
        theta = np.zeros(width.shape) # rotation angle of the rectangle

        rotation_class = np.zeros(theta.shape, dtype='int32') # rotation class of the rectangle
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        f_anno = open(anno_path, 'w')

        for i in range(theta.shape[0]):
            if center_x[i] > open_point_x[i]:
                theta[i] = np.arctan((open_point_y[i] - center_y[i]) / (center_x[i] - open_point_x[i])) * 180 / np.pi
            elif center_x[i] < open_point_x[i]:
                theta[i] = np.arctan((center_y[i] - open_point_y[i]) / (open_point_x[i] - center_x[i])) * 180 / np.pi
            else:
                theta[i] = -90.0
            rotation_class[i] = round((theta[i] + 90) / 10) + 1
            if x_min[i] < 0 or y_min[i] < 0 or x_max[i] > 1280 or y_max[i] > 720:
                continue
            f_anno.write('{} {} {} {} {}\n'.format(rotation_class[i], x_min[i], y_min[i], x_max[i], y_max[i]))

        f_anno.close()

        cv2.imwrite(img_path, img) # write the RGD image
        tmp_num = random.randint(1, 5)
        if (test_count >= test_num) or ((train_count < train_num) and (not tmp_num == 1)):
            is_training = True
        else:
            is_training = False
        if is_training:
            log_train_list.append(img_name)
            log_train_list.append('\n')
            train_total_number += 1
            train_count += 1
        else:
            log_test_list.append(img_name)
            log_test_list.append('\n')
            test_total_number += 1
            test_count += 1

        # if int(img_index) == 0:
            # print(grasp.shape)
            # tmp_mask = grasp[:, 0] > 720
            # print(grasp[tmp_mask].shape)
            # print(theta.shape)
            # print(theta[30:50])
            # print(rotation_class.shape)
            # print(rotation_class[40:70])
            # print(center_x.shape)
            # print(center_x[30:50])
            # print(center_y.shape)
            # print(center_y[30:50])
            # print(width.shape)
            # print(width[30:50])
            # print(height.shape)
            # print(height[30:50])
            # print(x_min.shape)
            # print(x_min[30:50])
            # print(y_min.shape)
            # print(y_min[30:50])
            # print(x_max.shape)
            # print(x_max[30:50])
            # print(y_max.shape)
            # print(y_max[30:50])
            # print(rgb_img.shape)
            # print(rgb_img[123][470:490])
            # print(b_img.shape)
            # print(b_img[654][970:990])
            # print(reshaped_g_img.shape)
            # print(reshaped_g_img[123][470:490])
            # print(reshaped_r_img.shape)
            # print(reshaped_r_img[123][470:490])
            # print(reshaped_depth_img.shape)
            # print(reshaped_depth_img[123][470:490])
            # print(img.shape)
            # print(img[123][470:490])
            # print(reshaped_depth_img.shape)
            # print(reshaped_depth_img[456][770:790])
            # print(img.shape)
            # print(img[456][770:790])
            # cv2.imwrite('1.png', img)
    # record the list
    f_train = open('../grasp_data/ImageSets/train.txt', 'a')
    f_train.writelines(log_train_list)
    f_train.close()
    f_test = open('../grasp_data/ImageSets/test.txt', 'a')
    f_test.writelines(log_test_list)
    f_test.close()
    print('scene {} done'.format(scene_index))
    print('{} images for training in scene {}'.format(train_count, scene_index))
    print('{} images for testing in scene {}'.format(test_count, scene_index))

f_image_skip.close()
f_scene_skip.close()

print('The number of images for training: {}'.format(train_total_number))
print('The number of images for testing: {}'.format(test_total_number))
print('------------------------------------------')
