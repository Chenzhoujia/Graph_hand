import sys
import numpy as np
from data.evaluation import Evaluation
xyz_gt_list = np.loadtxt("xyz_gt_list_all.txt")
xyz_val_list1 = np.loadtxt("/home/chen/Documents/GNN_demo/train_values_all_1.txt")
xyz_val_list1 = xyz_val_list1.reshape([-1,42])
xyz_val_list2 = np.loadtxt("/home/chen/Documents/GNN_demo/train_values_all_2.txt")
xyz_val_list2 = xyz_val_list2.reshape([-1,42])
xyz_val_list3 = np.loadtxt("/home/chen/Documents/GNN_demo/train_values_all_3.txt")
xyz_val_list3 = xyz_val_list3.reshape([-1,42])
xyz_val_list4 = np.loadtxt("/home/chen/Documents/GNN_demo/train_values_all_4.txt")
xyz_val_list4 = xyz_val_list4.reshape([-1,42])
xyz_val_list = np.concatenate((xyz_val_list1, xyz_val_list2,xyz_val_list3,xyz_val_list4), axis=0)

meanJntError = []
err_path = "/home/chen/Documents/denseReg-master/exp/train_cache/saved/GNN/_error.txt"
for i in range(8252):
    xyz_val = xyz_val_list[i,:]
    gt_val = xyz_gt_list[i,:]
    meanJntError.append(Evaluation.meanJntError(xyz_val, gt_val))
Evaluation.plotError(meanJntError, err_path)