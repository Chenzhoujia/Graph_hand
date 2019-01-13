#coding:UTF-8
import tensorflow as tf

#输入
input1 = tf.placeholder(tf.int32)#placeholder(dtype,shape),定义一个3行2列的矩阵
#输出
#output = tf.matmul(input1,input2)#matmul(),矩阵乘法
output = input1
#执行
with tf.Session() as sess:
    result = sess.run(output,feed_dict = {input1:1})
    print(result)