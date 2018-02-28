#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # filter the warning
import matplotlib as mplot
mplot.use("TkAgg")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt;

folderPath = '/home/vergil/Files/Git/FaceRecoginition/FaceDatabase/'

def addLayer(inputData,inSize,outSize,activity_function = None):  
    Weights = tf.Variable(tf.random_normal([inSize,outSize]))   
    basis = tf.Variable(tf.zeros([1,outSize])+0.1)    
    weights_plus_b = tf.matmul(inputData,Weights)+basis  
    if activity_function is None:  
        ans = weights_plus_b  
    else:  
        ans = activity_function(weights_plus_b)  
    return ans  


filePath = folderPath + 'att_faces_10_people/'

# parameter set
num_samples = 10 # Number of samples/people
num_images_each_sample = 10 # NUmber of photos of each sample
train_test_ratio = 0.6
num_train = int(train_test_ratio * num_images_each_sample); # Number of Trained data
num_test = int((1-train_test_ratio) * num_images_each_sample) # Number of Test data
num_train_images = int(num_samples * num_train)
num_test_images = int(num_samples * num_test)

imgSet_train = []
imgSet_test = []
labSet_train = []
labSet_test = []
imSize = [112, 92]




with tf.Session() as sess:


	# Build image set for train
	for i in range(1, num_train+1):
		for j in range(1, num_samples+1):

			imgPath = filePath + 's'+str(j)+'/'+str(i)+'.jpg'
			image_raw_data_jpg = tf.gfile.FastGFile(imgPath, 'r').read()
			img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
			img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
			img_data_jpg = tf.cast(img_data_jpg, tf.int32)
			# dtype is changed to tf.int32 instead of tf.uint8 cause not supported on this computer
			img_data_jpg = tf.reshape(img_data_jpg, [imSize[0]*imSize[1]])
			
			imgSet_train.append(img_data_jpg)

			label = [0,0,0,0,0,0,0,0,0,0]
			label[j-1] = 1

			labSet_train.append(label)

	labSet_train = tf.stack(labSet_train)


	# Build image set & label for test
	for i in range(num_train+1, num_images_each_sample+1):
		for j in range(1, num_samples+1):

			imgPath = filePath + 's'+str(j)+'/'+str(i)+'.jpg'
			image_raw_data_jpg = tf.gfile.FastGFile(imgPath, 'r').read()
			img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
			img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
			img_data_jpg = tf.cast(img_data_jpg, tf.int32)
			# dtype is changed to tf.int32 instead of tf.uint8 cause not supported on this computer
			img_data_jpg = tf.reshape(img_data_jpg, [imSize[0]*imSize[1]])
			
			imgSet_test.append(img_data_jpg)

			label = [0,0,0,0,0,0,0,0,0,0]
			label[j-1] = 1

			labSet_test.append(label)

	labSet_test = tf.stack(labSet_test)


	# Calculate mean image
	imgSet_mean = tf.add_n(imgSet_train)/num_train_images
	#print(imgSet_mean.eval())

	mean_face = tf.reshape(imgSet_mean, [112, 92])

	# plt.figure('Face1')
	# plt.imshow(mean_face.eval(), cmap=plt.cm.gray_r)
	# plt.show()

	# Use EIGENFACE method
	# Determine the difference matrix: A 
	A_temp = imgSet_train - np.tile(imgSet_mean, num_train_images)

	# The following steps solve the problem of tf.stack(A_temp,0)
	A = []
	for i in range(0, num_train_images):
		A.append(A_temp[i])
	A = tf.stack(A,0)
	A = tf.transpose(A)

	A = tf.cast(A, tf.float32)

	# Obtain the eigenvectors & eigenvalues of A

	[e, V] = tf.self_adjoint_eig(tf.matmul(tf.transpose(A), A))

	# if D[0][0] == 0, set D[0][0] to 1 for numerical resaon

	# Choose the best 95% of eigenvalues as the new reduced dimension
	p_eigenfaces = 0.95

	eigsum = tf.reduce_sum(e, 0)
	csum = 0;
	for i in range(num_train_images-1, 0, -1):
		csum = csum + e[i]
		r = csum/eigsum


		# Cost a lot to call .eval() every time
		if r.eval() > p_eigenfaces:
			k95 = i
			break
	print('The number of eigenvaluse is '+str(num_train_images))
	print('Keep the index from '+str(k95)+' to '+str(num_train_images-1))
	print('The last '+str(num_train_images-k95)+' are kept')

	# Determin the weights with reference to the set of eigenfaces Use
	# Use the last k95 componments
	i_start = k95
	i_end = num_train_images-1


	# Obtain the ranked eigenfaces Ur  ???not sure with transpose part
	Ur = tf.matmul(A, tf.transpose(V[:][i_start:i_end+1]))
	
	# Obtain the ranged eigenvalues 
	er = tf.diag(e[i_start:i_end+1])

	# Obtain the eigen weight mattrix
	EigenWeights = tf.matmul(tf.matmul(tf.matrix_inverse(er), tf.transpose(Ur)), A)


	# Build the model and train

	dinput = tf.transpose(EigenWeights)
	doutput = labSet_train

	# print(dinput[0])
	# print(doutput[0])

	tfinput = tf.placeholder(tf.float32, [num_train_images-k95, None])
	tfoutput = tf.placeholder(tf.float32, [num_samples, None])

	layer1 = addLayer(tfinput, 38, 15, activity_function=tf.nn.sigmoid)
	layer2 = addLayer(layer1, 15, 10, activity_function=tf.nn.sigmoid)

	