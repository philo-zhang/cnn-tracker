import os
import numpy as np
import cv2
import h5py
import random

dataPath = '/media/philo/1T_HardDisk/cnn_shift_data'
imgPath1 = dataPath + '/data1'
imgPath2 = dataPath + '/data2'
imgNum = 107611
trainNum = 80000
X1 = np.zeros((imgNum,3,64,64),dtype = 'float')
X2 = np.zeros((imgNum,3,64,64),dtype = 'float')
Y = np.zeros((imgNum,2),dtype = 'float')
rin = open(dataPath + '/labels.txt','r')
train_filename = dataPath + '/train.h5'
test_filename = dataPath + '/test.h5'
for i in range(0,imgNum):
    name1 = '{:05d}.jpg'.format(i)
    name2 = '{:05d}.jpg'.format(i)
    imgName1 = os.path.join(imgPath1,name1)
    imgName2 = os.path.join(imgPath2,name2)
    img1 = cv2.imread(imgName1,1)
    img2 = cv2.imread(imgName2,1)
    for j in range(3):
        X1[i,j] = img1[:,:,j]
        X2[i,j] = img2[:,:,j]
    line = rin.readline()
    nums = line.split(' ')
    for j in range(len(nums)):
        Y[i,j] = nums[j]
#cv2.imshow('img',img1);
#cv2.waitKey(0);
#cv2.imshow('img',img2);
#cv2.waitKey(0);
print Y[0,:]
X = np.zeros((imgNum*2,3,64,64),dtype='float')
X[0:imgNum] = X1
X[imgNum:imgNum*2] = X2
print X.shape
Xmean = np.mean(X, axis=0)
XmeanM = np.tile(Xmean, (imgNum,1,1,1))
X1 = X1-XmeanM
X1 = X1/255.0
X2 = X2-XmeanM
X2 = X2/255.0
del X, XmeanM

rand = np.array(range(0,trainNum))
random.shuffle(rand)
X1_train = np.zeros((trainNum,3,64,64),dtype='float')
X2_train = np.zeros((trainNum,3,64,64),dtype='float')
Y_train = np.zeros((trainNum,2),dtype='float')
X1_train = X1[rand]
X2_train = X2[rand]
Y_train = Y[rand]
X1_test = X1[trainNum:imgNum]
X2_test = X2[trainNum:imgNum]
Y_test = Y[trainNum:imgNum]
del X1, X2, Y

X_train = np.zeros((trainNum,6,64,64),dtype='float')
X_test = np.zeros((imgNum-trainNum,6,64,64),dtype='float')
X_train[:,0:3] = X1_train
X_train[:,3:6] = X2_train
X_test[:,0:3] = X1_test
X_test[:,3:6] = X2_test
del X1_train, X2_train, X1_test, X2_test

with h5py.File(train_filename, 'w') as f:
    f['data'] = X_train
    f['label'] = Y_train
with open(os.path.join(dataPath,'train.txt'), 'w') as f:
    f.write(train_filename + '\n')

with h5py.File(test_filename, 'w') as f:
    f['data'] = X_test
    f['label'] = Y_test
with open(os.path.join(dataPath, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')
with h5py.File(dataPath + '/mean.h5', 'w') as f:
    f['mean'] = Xmean
