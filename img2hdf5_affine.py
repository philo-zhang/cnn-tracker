import os
import numpy as np
import cv2
import h5py
import random
import math

dataPath = '/media/philo/1T_HardDisk/cnn_affine_data'
imgPath1 = dataPath + '/data1'
imgPath2 = dataPath + '/data2'
imgNum = len(os.listdir(imgPath1))
batchSize = 50000
batchNum = int(math.ceil(imgNum*1.0/batchSize))
trainBatchNum = batchNum*3/4
testBatchNum = batchNum-trainBatchNum
print batchNum, trainBatchNum, testBatchNum
randTrain = np.array(range(0, trainBatchNum*batchSize))
random.shuffle(randTrain)
randTest = np.array(range(trainBatchNum*batchSize, imgNum))
random.shuffle(randTest)
rand = np.concatenate((randTrain, randTest), 0)
print rand.shape
rin = open(dataPath + '/labels.txt','r')
Y = np.zeros((imgNum, 8), dtype = np.float32)
for i in range(0, imgNum):
    line = rin.readline()
    nums = line.split(' ')
    for j in range(len(nums)):
        Y[i,j] = nums[j]
Y = Y[rand]
rin.close()
Xmean1 = np.zeros((batchNum,3,64,64), dtype=np.float32)
Xmean2 = np.zeros((batchNum,3,64,64), dtype=np.float32)
for i in range(0,batchNum):
    start = i*batchSize
    end = (i+1)*batchSize
    if end>imgNum:
        end = imgNum
    X1 = np.zeros((end-start,3,64,64),dtype=np.int8)
    X2 = np.zeros((end-start,3,64,64),dtype=np.int8)
    for j in range(start,end):
        name1 = '{:06d}.jpg'.format(rand[j])
        name2 = '{:06d}.jpg'.format(rand[j])
        imgName1 = os.path.join(imgPath1,name1)
        imgName2 = os.path.join(imgPath2,name2)
        img1 = cv2.imread(imgName1,1)
        img2 = cv2.imread(imgName2,1)
        for k in range(3):
            X1[j-start,k] = img1[:,:,k]
            X2[j-start,k] = img2[:,:,k]
    Xmean1[i] = np.mean(X1, axis=0)
    Xmean2[i] = np.mean(X2, axis=0)
    y = Y[start:end,:]
    with h5py.File(dataPath + '/batch{:d}.h5'.format(i), 'w') as f:
        f['X1'] = X1
        f['X2'] = X2
        f['y'] = y
    print X1.shape, X2.shape, y.shape
Xmean = np.concatenate((Xmean1, Xmean2), 0)
Xmean = np.mean(Xmean, 0)
Xmean = np.reshape(Xmean,(1,)+Xmean.shape)
print Xmean.shape
for i in range(0, batchNum):
    with h5py.File(dataPath + '/batch{:d}.h5'.format(i), 'r') as r:
        X1 = np.array(r['X1'], dtype=np.float32)
        X2 = np.array(r['X2'], dtype=np.float32)
        y = np.array(r['y'], dtype=np.float32)
        X1 = (X1-Xmean)/255.0
        X2 = (X2-Xmean)/255.0
        X = np.concatenate((X1, X2), 1)
        print X.shape, y.shape
        del X1, X2
        if i<trainBatchNum:
            with h5py.File(dataPath + '/train_batch_{:d}.h5'.format(i), 'w') as w:
                w['data'] = X
                w['label'] = y
            with open(dataPath + '/train.txt', 'a') as f:
                f.write(dataPath + '/train_batch_{:d}.h5'.format(i) + '\n')
        else:
            with h5py.File(dataPath + '/test_batch_{:d}.h5'.format(i-trainBatchNum), 'w') as w:
                w['data'] = X
                w['label'] = y
            with open(dataPath + '/test.txt', 'a') as f:
                f.write(dataPath + '/test_batch_{:d}.h5'.format(i-trainBatchNum) + '\n')
        del X, y
with h5py.File(dataPath + '/mean.h5', 'w') as f:
    f['mean'] = Xmean

