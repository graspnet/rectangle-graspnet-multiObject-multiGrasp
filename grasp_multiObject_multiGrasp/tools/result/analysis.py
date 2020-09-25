import numpy as np

# FPPI.npy  miss_rate.npy  total_false_positive_num.npy  total_gt_num.npy  total_missed_gt_grasp_num.npy  total_proposed_num.npy

FPPI = np.load('FPPI.npy')
miss_rate = np.load('miss_rate.npy')
total_false_positive_num = np.load('total_false_positive_num.npy')
total_gt_num = np.load('total_gt_num.npy')
total_missed_gt_grasp_num = np.load('total_missed_gt_grasp_num.npy')
total_proposed_num = np.load('total_proposed_num.npy')

total_true_positive_num = total_proposed_num - total_false_positive_num
accuracy = total_true_positive_num / total_proposed_num

print('FPPI: {}'.format(FPPI))
print('miss rate: {}'.format(miss_rate))
print('accuracy: {}'.format(accuracy))