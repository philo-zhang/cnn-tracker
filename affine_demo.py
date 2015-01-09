import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import h5py
import cv2
import os
import time
import math

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

batchSize = 5
labelSize = 8

dataSet = '/media/philo/1T_HardDisk/dataset'
dataPath = '/media/philo/1T_HardDisk/cnn_affine_data'
seqName = 'Doll'
start = 575
end = 580
sequence = os.path.join(dataSet, seqName)
rin = open(sequence + '/groundtruth_rect.txt', 'r')
rect = np.zeros((4), dtype='float')
ratio = np.zeros((batchSize,2), dtype='float')

def pose_estimation(P1, P2):
    transM = np.zeros((P1.shape[0], 2, 3), dtype = float)
    for num in range(P1.shape[0]):
        p1 = P1[num]
        p1 = p1.reshape(4,2)
        p1 = np.transpose(p1)
        center = (p1[:,0:1]+p1[:,2:3])/2
        p1 = p1-np.tile(center,(1,4))
        p1 = np.concatenate((p1, np.ones((1,4))), 0)
        p2 = P2[num]
        p2 = p2.reshape(4,2)
        p2 = np.transpose(p2)
        p2 = p2-np.tile(center,(1,4))
        H = np.dot(p2, linalg.pinv(p1))
        HSub = H[0:2,0:2].copy()
        A = math.sqrt(sum((HSub**2).flatten(1))/2)
        HSub = HSub/A
        U, S, Vh = linalg.svd(HSub)
        R = np.dot(U,Vh)
        D = H[:,2:3].copy()
        transM[num] = np.concatenate((R*A,D),1)
    return transM
       
for imgNum in range(1,end+1):
    line = rin.readline()
    nums = line.split(' ')
    if imgNum == start:
        for j in range(len(nums)):
            rect[j] = nums[j]
        img = cv2.imread(sequence + '/img/{:04d}.jpg'.format(imgNum), 1)
        imcrop = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        imcrop = cv2.resize(imcrop,(64,64))
        for i in range(batchSize):
            cv2.imwrite(dataPath + '/testData/data1/{:d}.jpg'.format(i), imcrop)
        cv2.imwrite(dataPath + '/testData/imgs/{:d}.jpg'.format(imgNum), img)
    if imgNum >start:
        img = cv2.imread(sequence + '/img/{:04d}.jpg'.format(imgNum), 1)
        imcrop = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        ratio[imgNum-start-1] = np.array([rect[2], rect[3]])/64.0
        #print 'ratio=',ratio
        imcrop = cv2.resize(imcrop,(64,64))
        cv2.imwrite(dataPath + '/testData/data2/{:d}.jpg'.format(imgNum-start-1), imcrop)
        cv2.imwrite(dataPath + '/testData/imgs/{:d}.jpg'.format(imgNum), img)

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

modelFile = 'train_val_affine_deploy.prototxt'
pretrainedFile = 'affine_iter_20000.caffemodel'

net = caffe.Net(modelFile, pretrainedFile)
net.set_phase_test()
net.set_mode_gpu()

imgPath1 = dataPath + '/testData/data1'
imgPath2 = dataPath + '/testData/data2'
X1 = np.zeros((batchSize,3,64,64), dtype = 'float')
X2 = np.zeros((batchSize,3,64,64), dtype = 'float')
with h5py.File(dataPath + '/mean.h5','r') as f:
    Xmean = np.array(f['mean'])
#print 'Xmean.shape=', Xmean.shape
XmeanM = np.tile(Xmean, (batchSize,1,1,1))
#print 'XmeanM.shape=', XmeanM.shape

vertex = np.array([rect[0],rect[1],rect[0]+rect[2],rect[1],rect[0]+rect[2],rect[1]+rect[3],rect[0],rect[1]+rect[3]])
vertices = np.tile(vertex,(batchSize,1))

for iter in range(1):
    for i in range(0,batchSize):
        name1 = '{:d}.jpg'.format(i)
        name2 = '{:d}.jpg'.format(i)
        imgName1 = os.path.join(imgPath1, name1)
        imgName2 = os.path.join(imgPath2, name2)
        img1 = cv2.imread(imgName1, 1)
        img2 = cv2.imread(imgName2, 1)
        for j in range(3):
            X1[i,j] = img1[:,:,j]
            X2[i,j] = img2[:,:,j]
    X1 = (X1-XmeanM)/255.0
    X2 = (X2-XmeanM)/255.0
    X = np.zeros((batchSize,6,64,64), dtype = 'float')
    X[:,0:3] = X1
    X[:,3:6] = X2
    #print 'X.shape=', X.shape
    out = net.forward_all(data=X)
    output = out['fc7']
    output = output.reshape(*(output.shape[0:2]))
    ratio = np.tile(ratio, (1,4))
    verticesNew = vertices - np.multiply(output, ratio)
    transM = pose_estimation(vertices, verticesNew)
    #print 'tranM=', transM
    verticesTrue = np.zeros((batchSize, 2, 4), dtype = float)
    for i in range(batchSize):
        vertex = vertices[i]
        vertex = vertex.reshape(4, 2)
        vertex = vertex.transpose()
        center = (vertex[:,0:1]+vertex[:,2:3])/2
        vertex = vertex-np.tile(center,(1,4))
        vertex = np.concatenate((vertex, np.ones((1,4))), 0)
        verticesTrue[i] = np.dot(transM[i], vertex)
        verticesTrue[i] = verticesTrue[i]+np.tile(center,(1,4))

plt.subplots(figsize=(12,20))
colordef = 'rgbycmkwmk'
for i in range(6):
    ax = plt.subplot(10,6,i+1)
    im = plt.imread(dataPath + '/testData/imgs/{:d}.jpg'.format(i+start))
    plt.imshow(np.array(im))
    plt.axis('off')
    plt.title('frame '+str(i+start), fontsize=8)

for i in range(3):
    for j in range(2):
        ax = plt.subplot2grid((10,6),(3*i+1,3*j),colspan=3,rowspan=3)
        im = plt.imread(dataPath + '/testData/imgs/{:d}.jpg'.format(2*i+j+start))
        plt.imshow(np.array(im))
        if 2*i+j > 0:
            vertex = verticesNew[2*i+j-1]
            vertex = np.reshape(vertex, (4,2))
            vertex = vertex[[0,1,2,3,0],:]
            vertex = np.transpose(vertex)
            vertexTrue = verticesTrue[2*i+j-1]
            vertexTrue = vertexTrue[:,[0,1,2,3,0]]
            #print 'vertex=', vertex
            #print 'vertexTrue=', vertexTrue
            after, = plt.plot(vertex[0],vertex[1],label='after prediction',color='blue',linewidth=1.2)
            pose, = plt.plot(vertexTrue[0],vertexTrue[1],label='after pose-estimation',color='red',linewidth=1.2)
        xcoor = [rect[0],rect[0]+rect[2],rect[0]+rect[2],rect[0],rect[0]]
        ycoor = [rect[1],rect[1],rect[1]+rect[3],rect[1]+rect[3],rect[1]]
        before, = plt.plot(xcoor,ycoor,label='before prediction',color='green',linewidth=2)
        plt.axis('off')
        if 2*i+j > 0:
            plt.legend(handles=[before, after, pose], prop={'size':'10'})
        plt.title('frame {:d}'.format(2*i+j+start), fontsize=30)
plt.draw()
plt.savefig('affine.jpg', bbox_inches='tight')
