# coding=UTF-8
from __future__ import print_function, absolute_import, division

#import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
from datetime import datetime
import pickle
from base_graph import base_graph
FLAGS = tf.app.flags.FLAGS

def _average_gradients(tower_grads):
    '''calcualte the average gradient for each shared variable across all towers on multi gpus
    Args:
        tower_grads: list of lists of (gradient, variable) tuples. len(tower_grads)=#tower, len(tower_grads[0])=#vars 
    Returns:
        List of paris (gradient, variable) where the gradients has been averaged across
        all towers
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # over different variables
        grads = []
        for g, _ in grad_and_vars:
            # over different towers
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(model, restore_step=None):
    '''train the provided model
    model: provide several required interface to train
    '''
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.train.exponential_decay(model.init_lr,
                                       global_step,
                                       model.decay_steps,
                                       model.lr_decay_factor,
                                       staircase=True)

        print('[train] learning rate decays per %d steps with rate=%f'%(
            model.decay_steps,model.lr_decay_factor))
        print('[train] initial learning_rate = %f'%model.init_lr)
        tf.summary.scalar('learning_rate', lr)
        opt = model.opt(lr)
        #构建训练batch
        batches = model.batch_input(model.train_dataset)

        # getdata chen_begin
        loss, gt_poses_v, xyz_pts_v = model.loss(*batches)
        # getdata chen_end

        tf.summary.scalar('loss', loss)

        if model.is_validate:
            # set batch_size as 3 since tensorboard visualization
            val_batches = model.batch_input(model.val_dataset, 3)
            model.test(*val_batches) # don't need the name

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        accu_steps = float(FLAGS.sub_batch)

        grads = opt.compute_gradients(loss)
        accum_grads = []
        for grad, var in grads:
            if grad is not None:
                accum_grads.append(tf.Variable(tf.zeros_like(grad), trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    name=var.op.name+'_accu_grad'))
            else:
                accum_grads.append(tf.Variable(tf.zeros_like(var), trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    name=var.op.name+'_accu_grad'))

        reset_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_grads]
        accum_op = [accum_grads[i].assign_add(grad[0]) for i, grad in enumerate(grads)if grad[0] is not None]

        ave_grad = [(tf.clip_by_value(tf.divide(accum_grads[i], accu_steps), -0.2, 0.2),
                     grad[1]) for i, grad in enumerate(grads)]
        apply_gradient_op = opt.apply_gradients(ave_grad, 
                                                global_step=global_step)

        for ave_grad, grad_and_var in zip(ave_grad, grads):
            grad, var = grad_and_var[0], grad_and_var[1]
            if grad is not None:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name+'/gradients', ave_grad)

        # variable_averages = tf.train.ExponentialMovingAverage(
            # model.moving_average_decay, global_step)
        # variables_to_average = tf.trainable_variables()
        # var_1, var_2 = tf.moving_average_variables()[0], tf.moving_average_variables()[1]
        # variable_averages_op = variable_averages.apply(variables_to_average)

        batchnorm_update_op = tf.group(*batchnorm_updates)
        # group all training operations into one 
        # train_op = tf.group(apply_gradient_op, variable_averages_op)
        train_op = tf.group(apply_gradient_op)

        # add_var_list chen_begin  如果网络使用了nolocal，网络结构改变了，在加载预训练参数的时候需要生成新的saver_part，而保存的时候则需要全部的
        if FLAGS.nolocal_UM:
            lst_vars = []
            for v in tf.global_variables():
                if 'no_local' in v.name:
                    pass
                else:
                    lst_vars.append(v)
                    print(v.name, 'load....')
            saver_part = tf.train.Saver(var_list=lst_vars)
        saver = tf.train.Saver(tf.global_variables())

        # add_var_list chen_end
        summary_op = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)))

        sess.run(init_op)
        start_step = 0
        # to resume the training
        if restore_step is not None:# and restore_step>0:
            checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%restore_step)
            if FLAGS.nolocal_UM:
                saver_part.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, checkpoint_path)
            #restore_model(sess, "/home/chen/Documents/denseReg-master/exp/train_cache/nyu_training_s2_f128_daug_um_v1/model.ckpt-7600")
            start_step = restore_step

        tf.train.start_queue_runners(sess=sess)

        #TODO: change to tf.train.SummaryWriter()
        summary_writer = tf.summary.FileWriter(
            model.summary_dir,
            graph=sess.graph)

        # finally into the training loop
        print('finally into the long long training loop')

        log_path = os.path.join(model.train_dir, 'training_log.txt')
        f = open(log_path, 'a')
        # getdata chen_begin
        graph_path = os.path.join(model.train_dir, 'graph.dat')
        f_graph_path = open(graph_path, 'ab')
        start_step = 0
        for step in range(start_step, 14551*2):#1:15(sub_step*batch_size)->14551
        # getdata chen_end

        #for step in range(start_step, model.max_steps):
            if f.closed:
                f = open(log_path, 'a')
            # getdata chen_begin
            if f_graph_path.closed:
                f_graph_path = open(graph_path, 'ab')
            # getdata chen_end
            start_time = time.time()
            ave_loss = 0
            sess.run(reset_op)
            for sub_step in range(int(accu_steps)):
                #_, _, loss_value= sess.run([accum_op, batchnorm_update_op])
                # getdata chen_begin
                gt_poses_vv, xyz_pts_vv = sess.run([gt_poses_v, xyz_pts_v])
                loss_value = 0
                print("write>>" + str(step))
                for b_num in range(FLAGS.batch_size):
                    pickle.dump(base_graph(xyz_pts_vv[b_num, :],gt_poses_vv[b_num,:]), f_graph_path)
                print("done>>")
                # getdata chen_end
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                ave_loss += loss_value

            _ = sess.run([train_op])
            ave_loss /= accu_steps
            duration = time.time() - start_time

            if step%5 == 0:
                format_str = '[model/train_multi_gpu] %s: step %d/%d, loss = %.3f, %.3f sec/batch, %.3f sec/sample'
                print(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps)))
                f.write(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps))+'\n')
                f.flush()

            if step%20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


            if step%40 == 0 and hasattr(model, 'do_test'):
                pass
                model.do_test(sess, summary_writer, step)

            if step%100 == 0 or (step+1) == model.max_steps:
                if not os.path.exists(model.train_dir):
                    os.makedirs(model.train_dir)
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print('model has been saved to %s\n'%checkpoint_path)
                f.write('model has been saved to %s\n'%checkpoint_path)
                f.flush()

        print('finish train')
        f.close()
def restore_model(sess, path):
    print("Restoring weights from file %s." % path)
    with open(path, 'rb') as in_file:
        data_to_load = pickle.load(in_file)

    # Assert that we got the same model configuration
    """
    assert len(self.params) == len(data_to_load['params'])
    for (par, par_value) in self.params.items():
        # Fine to have different task_ids:
        if par not in ['task_ids', 'num_epochs']:
            assert par_value == data_to_load['params'][par]
    """
    variables_to_initialize = []
    with tf.name_scope("restore"):
        restore_ops = []
        used_vars = set()
        for variable in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            used_vars.add(variable.name)
            if variable.name in data_to_load['weights']:
                restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
            else:
                print('Freshly initializing %s since no saved value was found.' % variable.name)
                variables_to_initialize.append(variable)
        for var_name in data_to_load['weights']:
            if var_name not in used_vars:
                print('Saved weights for %s not used by model.' % var_name)
        restore_ops.append(tf.variables_initializer(variables_to_initialize))
        sess.run(restore_ops)

