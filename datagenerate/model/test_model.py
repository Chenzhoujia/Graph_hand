# coding=UTF-8
from __future__ import print_function, absolute_import, division
import sys
sys.path.append(r'/home/chen/Documents/denseReg-master')
#import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
import cv2
from datetime import datetime
from data.evaluation import Evaluation
from data.visualization import figure_joint_skeleton2

FLAGS = tf.app.flags.FLAGS

def test(model, selected_step):
    with tf.Graph().as_default():
        total_test_num = model.val_dataset.exact_num
        #获得原始数据，唯一的处理就是把深度图crop了一下
        dms, poses, cfgs, coms, names = model.batch_input_test(model.val_dataset)
        model.test(dms, poses, cfgs, coms, reuse_variables=None)

        # dms, poses, names = model.batch_input_test(model.val_dataset)
        # model.test(dms, poses, reuse_variables=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if selected_step is not None:
            checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%selected_step)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, checkpoint_path)
            print('[test_model]model has been resotored from %s'%checkpoint_path)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            model.summary_dir+'_'+model.val_dataset.subset,
            graph=sess.graph)
        
        res_path = os.path.join(model.train_dir, '%s-%s-result'%(model.val_dataset.subset, datetime.now()))
        res_path = res_path.replace(' ', '_')

        res_txt_path = res_path+'.txt'
        if os.path.exists(res_txt_path):
            os.remove(res_txt_path)
        err_path = res_path+'_error.txt'
        f = open(res_txt_path, 'w')

        # res_vid_path = res_path+'.avi'
        # codec = cv2.cv.CV_FOURCC('X','V','I','D')
        # the output size is defined by the visualization tool of matplotlib
        # vid = cv2.VideoWriter(res_vid_path, codec, 25, (640, 480))
        
        print('[test_model]begin test')
        test_num = 0
        step = 0
        meanJntError = []
        xyz_val_list = []
        xyz_gt_list = []
        while True:
            start_time = time.time()
            try:
                gt_vals, xyz_vals, name_vals,val_dms_v, gt_uvd_pts_v, uvd_pts_v = model.do_test(sess, summary_writer, step, names)
            except tf.errors.OutOfRangeError:
                print('run out of range')
                break

            duration = time.time()-start_time
            
            for xyz_val, gt_val, name_val,val_dms_v1, gt_uvd_pts_v1, uvd_pts_v1 in zip(xyz_vals, gt_vals, name_vals, val_dms_v, gt_uvd_pts_v, uvd_pts_v):
                #maxJntError.append(Evaluation.maxJntError(xyz_val, gt_val))

                #figure_joint_skeleton2(np.squeeze(val_dms_v1),gt_uvd_pts_v1,test_num,"gt")#.savefig("/home/chen/Documents/denseReg-master/exp/train_cache/saved/pretrain/image/"+str(test_num).zfill(7)+"_gt.png")
                #figure_joint_skeleton2(np.squeeze(val_dms_v1), uvd_pts_v1,test_num,"pt")#.savefig("/home/chen/Documents/denseReg-master/exp/train_cache/saved/pretrain/image/"+str(test_num).zfill(7)+"_pt.png")

                meanJntError.append(Evaluation.meanJntError(xyz_val, gt_val))
                xyz_val_list.append(xyz_val[np.newaxis, :])
                xyz_gt_list.append(gt_val[np.newaxis, :])
                xyz_val = xyz_val.tolist()
                res_str = '%s\t%s\n'%(name_val, '\t'.join(format(pt, '.4f') for pt in xyz_val))
                res_str = res_str.replace('/', '\\')
                f.write(res_str)
                # vid.write(vis_val)
                test_num += 1
                if test_num >= total_test_num:
                    xyz_gt_list = np.concatenate(xyz_gt_list, axis=0)
                    np.savetxt("a_5.txt", xyz_gt_list,fmt="%.5f")
                    np.savetxt("a_all.txt", xyz_gt_list)  # 缺省按照'%.18e'格式保存数据，以空格分隔
                    #xyz_val_list2 = np.loadtxt("a.txt")
                    print('finish test')
                    f.close()
                    Evaluation.plotError(meanJntError, err_path)

                    return
            f.flush()
            
            if step%101 == 0:
                print('[%s]: %d/%d computed, with %.2fs'%(datetime.now(), step, model.max_steps, duration))

            step += 1

