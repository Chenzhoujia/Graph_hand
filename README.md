# Graph_hand
一个经过测试,可使用的基于denseRge和graph_net的用图网络优化姿态估计结果的工程
# 图网络工程闭环：
# （1）：训练数据生成<---datagenerate

(a)nyu.py p26行，数据导入
directory = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/NYU/'

(b)model.train_dir，数据输出（用graph组织）

(c)写数据的工作在 train_single_gpu.py

运行：/home/chen/Documents/denseReg-master/model/hourglass_um_crop_tiny.py
参数：--dataset nyu --batch_size 3 --num_stack 2 --num_fea 128 --debug_level 2 --is_train True

(d)网络结构改变
hourglass_um_crop_tinp.py P66-70   388-391
chen_begin modify flags
tf.app.flags.DEFINE_boolean('nolocal_UM', False,
                            'nolocal_UM')
tf.app.flags.DEFINE_boolean('shape_aware', False,
                            'shape_aware')

2D_mask chen_begin
if FLAGS.shape_aware:
    gt_hms = gt_hms*mask
2D_mask chen_end
um_v1.py  P174-178

#no-local_chen_begin
if FLAGS.nolocal_UM:
    with tf.variable_scope('no_local'+str(i)):
        um_out = no_local(um_out)
#no-local chen_end

(e)模型参数加载

add_var_list chen_begin  如果网络使用了nolocal，网络结构改变了，在加载预训练参数的时候需要生成新的saver_part，而保存的时候则需要全部的
if FLAGS.nolocal_UM:
    lst_vars = []
    for v in tf.global_variables():
        if 'no_local' in v.name:
            pass
        else:
            lst_vars.append(v)
            print(v.name, 'load....')
    saver_part = tf.train.Saver(var_list=lst_vars)
if FLAGS.nolocal_UM:
    saver_part.restore(sess, checkpoint_path)
else:
    saver.restore(sess, checkpoint_path)


# （2）：图网络训练<----GNN_optim
通过saved文件夹下的bash脚本运行，训练好的模型也存在里面，上一个步骤生成的数据也应该放在这个文件夹下
https://github.com/deepmind/graph_nets
# （3）：dense master 待优化结果生成<---datagenerate
用测试模式运行
运行：/home/chen/Documents/denseReg-master/model/hourglass_um_crop_tiny.py
参数：--dataset nyu --batch_size 3 --num_stack 2 --num_fea 128 --debug_level 2 --is_train False
Graph_hand/datagenerate/model/test_model.py 中93～94写在文本文件中

# （4）：图网络优化<----GNN_opt_test
与训练时的模型保持一致
手动运行hand_GNN_test.py4次
因为GPU一次吃不了8252的图，所以分了4次，核心代码在hand_GNN_test中，生成数据文件在同一个文件目录下
# （5）：结果评估<---datagenerate
Graph_hand/datagenerate/model/test.py
