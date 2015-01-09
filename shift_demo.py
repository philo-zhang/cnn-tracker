import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import os
import time

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def imgCrop(dataPath, prediction):
    imgsPath = dataPath + '/testData/imgs'
    with h5py.File(dataPath + '/testData/' + prediction, 'r') as f:
        rects = np.array(f['rects'])
    imgName = imgsPath + '/{:04d}.jpg'.format(2628)
    img = cv2.imread(imgName)
    for i in range(10):
        imgName = dataPath + '/testData/data2/{:05d}.jpg'.format(i)
        imcrop = img[rects[i,1]:rects[i,1]+rects[i,3],rects[i,0]:rects[i,0]+rects[i,2]]
        imcrop = cv2.resize(imcrop,(64,64))
        cv2.imwrite(imgName, imcrop)

dataPath = '/media/philo/1T_HardDisk/cnn_shift_data'
Y = np.zeros((10,10), dtype='float')
rin = open(dataPath + '/testData/groundtruth.txt', 'r')
for i in range(10):
    line = rin.readline()
    nums = line.split(' ')
    for j in range(len(nums)):
        Y[i,j] = nums[j]
print 'Y = ', Y
rects = Y[:,0:4].copy()

with h5py.File(dataPath + '/testData/prediction0.h5', 'w') as f:
    f['rects'] = rects
plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

modelFile = 'train_val_deploy.prototxt'
pretrainedFile = 'train_iter_10000.caffemodel'

net = caffe.Net(modelFile, pretrainedFile)
net.set_phase_test()
net.set_mode_gpu()

imgPath1 = dataPath + '/testData/data1'
imgPath2 = dataPath + '/testData/data2'
X1 = np.zeros((10,3,64,64), dtype = 'float')
X2 = np.zeros((10,3,64,64), dtype = 'float')
with h5py.File(dataPath + '/mean.h5','r') as f:
    Xmean = np.array(f['mean'])
print 'Xmean.shape=', Xmean.shape
XmeanM = np.tile(Xmean, (10,1,1,1))
print 'XmeanM.shape=', XmeanM.shape

for iter in range(5):
    imgCrop(dataPath, 'prediction{:d}.h5'.format(iter))
    for i in range(0,10):
        name1 = '{:05d}.jpg'.format(i)
        name2 = '{:05d}.jpg'.format(i)
        imgName1 = os.path.join(imgPath1, name1)
        imgName2 = os.path.join(imgPath2, name2)
        img1 = cv2.imread(imgName1, 1)
        img2 = cv2.imread(imgName2, 1)
        for j in range(3):
            X1[i,j] = img1[:,:,j]
            X2[i,j] = img2[:,:,j]
    X1 = (X1-XmeanM)/255.0
    X2 = (X2-XmeanM)/255.0
    X = np.zeros((10,6,64,64), dtype = 'float')
    X[:,0:3] = X1
    X[:,3:6] = X2
    print 'X.shape=', X.shape
    out = net.forward_all(data=X)
    output = out['fc1']
    output = output.reshape(*(output.shape[0:2]))
    print output
    rects[:,0:2] -= np.multiply(output, Y[:,4:6]/(2**0))
    with h5py.File(dataPath + '/testData/prediction{:d}.h5'.format(iter+1), 'w') as f:
        f['rects'] = rects
plt.subplots(figsize=(10,16))
colordef = 'rgbycmkwmk'
for i in range(10):
    ax = plt.subplot(16,10,i+1)
    im = plt.imread(dataPath + '/testData/imgs/{:04d}.jpg'.format(i+2618))
    plt.imshow(np.array(im))
    rect = plt.Rectangle((Y[i,0],Y[i,1]), Y[i,2], Y[i,3], facecolor='none', edgecolor=colordef[i], linewidth=1.0)
    ax.add_artist(rect)
    plt.axis('off')
    plt.title('frame '+str(i+2618), fontsize=8)

for i in range(3):
    for j in range(2):
        ax = plt.subplot2grid((16,10),(5*i+1,5*j),colspan=5,rowspan=5)
        im = plt.imread(dataPath + '/testData/imgs/{:04d}.jpg'.format(2628))
        plt.imshow(np.array(im))
        with h5py.File(dataPath + '/testData/prediction{:d}.h5'.format(2*i+j)) as f:
            rects = np.array(f['rects'])
        print rects.shape, len(colordef)
        for k in range(10):
            rect = plt.Rectangle((rects[k,0],rects[k,1]), rects[k,2], rects[k,3], facecolor='none', edgecolor=colordef[k], linewidth=1.0)
            ax.add_artist(rect)
        plt.axis('off')
        plt.title('frame 2628 after iteration {:d}'.format(2*i+j), fontsize=15)
plt.draw()
plt.savefig('output.jpg', bbox_inches='tight')
