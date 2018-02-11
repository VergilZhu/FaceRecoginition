#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np

def addLayer(inputData,inSize,outSize,activity_function = None):  
    Weights = tf.Variable(tf.random_normal([inSize,outSize]))   
    basis = tf.Variable(tf.zeros([1,outSize])+0.1)    
    weights_plus_b = tf.matmul(inputData,Weights)+basis  
    if activity_function is None:  
        ans = weights_plus_b  
    else:  
        ans = activity_function(weights_plus_b)  
    return ans  
  
  
# x_data = np.linspace(-1,1,300)[:,np.newaxis] # 转为列向量  
# noise = np.random.normal(0,0.05,x_data.shape)  
# y_data = np.square(x_data)+0.5+noise

# xs = tf.placeholder(tf.float32,[None,1]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入  
# ys = tf.placeholder(tf.float32,[None,1])  
  
# l1 = addLayer(xs,1,10,activity_function=tf.nn.relu) # relu是激励函数的一种  
# l2 = addLayer(l1,10,1,activity_function=None)  
# loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-l2)),reduction_indices = [1]))#需要向相加索引号，redeuc执行跨纬度操作  
  
# train =  tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 选择梯度下降法  
  
# init = tf.initialize_all_variables()  
# sess = tf.Session()  
# sess.run(init)  
  
# for i in range(10000):  
#     sess.run(train,feed_dict={xs:x_data,ys:y_data}) 
#     if i%50 == 0:  
#         print sess.run(loss,feed_dict={xs:x_data,ys:y_data})   

_data_train = np.array([[1,6],[2,5],[3,7],[4,5],[5,3],[6,1],[7,2],[8,4],[9,2],[10,5]])
_lable_train = np.array([0,0,0,0,1,1,1,1,1,1]).reshape(-1,1)
# _data_test = [[1.5,3],[2.5,2],[3.5,4],[4.5,2]]
# _lable_test = [0,0,0,1]

xs = tf.placeholder(tf.float32,[None,2])
ys = tf.placeholder(tf.float32,[None,1])

hidden_layer = addLayer(xs, 2, 5, activity_function=tf.nn.sigmoid)
output_layer = addLayer(hidden_layer, 5, 1, activity_function=tf.nn.sigmoid)

loss = tf.reduce_sum(tf.square(ys-output_layer))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

for i in range(500):
	sess.run(train_step, feed_dict={xs:_data_train, ys:_lable_train})

	#print(sess.run(loss, feed_dict={xs:_data_train, ys:_lable_train}))

print('---------------')
print(sess.run(output_layer, feed_dict={xs:[[4,-1],[4,0],[4,1],[4,2],[4,3],[4,4],[4,5]]}))	