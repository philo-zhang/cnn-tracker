import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import h5py
import cv2
import os
import time
import math
import sys
import caffe

caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')

#batchSize = 5
templateSize = 10
sampleSize = 1
labelSize = 8
PI = 3.1415926

dataSet = '/media/philo/1T_HardDisk/dataset'
dataPath = '/media/philo/1T_HardDisk/cnn_affine_data'
class winParam:
    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle

def pose_estimation(P1, P2):
    print P1.shape, P2.shape
    transM = np.zeros((P1.shape[0], 2, 3), dtype = float)
    dAngle = np.zeros((P1.shape[0], 1), dtype = float)
    dSize = np.zeros((P1.shape[0], 1), dtype = float)
    dCenter = np.zeros((P1.shape[0], 2), dtype = float)
    for num in range(P1.shape[0]):
        p1 = P1[num]
        p1 = p1.reshape(4,2)
        p1 = np.transpose(p1)
        center = (p1[:,0:1]+p1[:,2:3])/2.0
        p1 = p1-np.tile(center,(1,4))
        p1 = np.concatenate((p1, np.ones((1,4))), 0)
        p2 = P2[num]
        p2 = p2.reshape(4,2)
        p2 = np.transpose(p2)
        p2 = p2-np.tile(center,(1,4))
        H = np.dot(p2, linalg.pinv(p1))
        HSub = H[0:2,0:2].copy()
        A = math.sqrt(sum((HSub**2).flatten(1))/2.0)
        HSub = HSub/A
        U, S, Vh = linalg.svd(HSub)
        R = np.dot(U,Vh)
        D = H[:,2:3].copy()
        dAngle[num] = math.acos(R[0,0])
        dSize[num] = A
        dCenter[num] = D.transpose()
        transM[num] = np.concatenate((R*A,D),1)
    return dCenter, dSize, dAngle, transM

def sampling(win, sampleSize):
    xSamples = np.rint(np.random.normal(win.center[0], 3, sampleSize)).astype(int)
    ySamples = np.rint(np.random.normal(win.center[1], 3, sampleSize)).astype(int)
    angleSamples = np.random.normal(win.angle, 1, sampleSize)
    widthSamples = np.rint(np.random.normal(win.size[0], 1, sampleSize)).astype(int)
    heightSamples = np.rint(np.random.normal(win.size[1], 1, sampleSize)).astype(int)
    winSample = winParam((0,0), 0, 0)
    winSamples = []
    winWeights = []
    for i in range(sampleSize):
        winSample.center = (xSamples[i], ySamples[i])
        winSample.angle = angleSamples[i]
        winSample.size = (widthSamples[i], heightSamples[i])
        winSamples.append(winSample)
        #draw(videoWri, imgName, imgNum, winSample, lost)
        #cv2.waitKey(0)
    return winSamples
    
def prediction(net, imgName, win, templates, weights):
    winSamples = sampling(win, sampleSize)
    winSamples = []
    winSamples.append(win)
    img = cv2.imread(imgName, 1)
    #cv2.imshow('img',img)
    candidates = []
    ratios = []
    for k in range(sampleSize):
        affMat = cv2.getRotationMatrix2D(tuple(winSamples[k].center), winSamples[k].angle, 1.0)
        rotatedSrc = cv2.warpAffine(img, affMat, (img.shape[1],img.shape[0]))
        imcrop = cv2.getRectSubPix(rotatedSrc, tuple(winSamples[k].size), tuple(winSamples[k].center))
        #cv2.imshow('rotated', rotatedSrc)
        #cv2.imshow('crop',imcrop)
        #cv2.waitKey(0)
        imcrop = cv2.resize(imcrop,(64,64))
        ratio = (winSamples[k].size[0]/64.0, winSamples[k].size[1]/64.0)
        candidate = np.zeros((3,64,64), dtype = int)
        for k in range(3):
            candidate[k] =imcrop[:,:,k]
        for i in range(len(templates)):
            candidates.append(candidate)
            ratios.append(ratio)

    templates = np.tile(templates, (sampleSize, 1, 1, 1))

    with h5py.File(dataPath + '/mean.h5','r') as f:
        Xmean = np.array(f['mean'])
    XmeanM = np.tile(Xmean, (sampleSize*templateSize,1,1,1))

    vertices = np.zeros((sampleSize*templateSize,8), dtype = float)
    for k in range(sampleSize):
        vertices[k*templateSize:(k+1)*templateSize] = np.array([winSamples[k].center[0]-winSamples[k].size[0]/2, winSamples[k].center[1]-winSamples[k].size[1]/2, winSamples[k].center[0]+winSamples[k].size[0]/2, winSamples[k].center[1]-winSamples[k].size[1]/2, winSamples[k].center[0]+winSamples[k].size[0]/2, winSamples[k].center[1]+winSamples[k].size[1]/2, winSamples[k].center[0]-winSamples[k].size[0]/2, winSamples[k].center[1]+winSamples[k].size[1]/2])
    temp1 = np.zeros((64,64,3),dtype = int)
    temp2 = np.zeros((64,64,3),dtype = int)
    for k in range(3):
        temp1[:,:,k] = templates[0][k]
        temp2[:,:,k] = candidates[0][k]

    #cv2.imshow('template',temp1)
    #cv2.imshow('candidate',temp2)
    templates = (templates-XmeanM)/255.0
    candidates = (candidates-XmeanM)/255.0
    print templates.shape, candidates.shape
    X = np.concatenate((templates, candidates), 1)
    print 'X.shape=', X.shape
    out = net.forward_all(data=X)
    output = out['fc7']
    output = output.reshape(*(output.shape[0:2]))
    ratios = np.tile(ratios, (1,4))
    verticesNew = vertices - np.multiply(output, ratios)
    #print vertices, verticesNew
    winNew = winParam((0,0),(0,0),0)
    vertex = np.array([win.center[0]-win.size[0]/2, win.center[1]-win.size[1]/2, win.center[0]+win.size[0]/2, win.center[1]-win.size[1]/2, win.center[0]+win.size[0]/2, win.center[1]+win.size[1]/2, win.center[0]-win.size[0]/2, win.center[1]+win.size[1]/2])

    dCenter, dSize, dAngle, transM = pose_estimation(vertices, verticesNew)
    #print 'output:', output
    #print 'verticesNew:', verticesNew
    #print 'transM:', transM
    #print 'dcenter:', dCenter
    print 'dSize:', dSize
    #print 'dAngle:', dAngle
    #print weights
    for i in range(sampleSize):
        for j in range(templateSize):
            winNew.center += (winSamples[i].center+dCenter[i*templateSize+j])*weights[i*templateSize+j]
            winNew.size += (winSamples[i].size*dSize[i*templateSize+j])*weights[i*templateSize+j]
            winNew.angle += (winSamples[i].angle+dAngle[i*templateSize+j])*weights[i*templateSize+j]
            #draw(videoWri, imgName, imgNum, winNew, lost)
            #cv2.waitKey(0)
    winNew.center = (int(round(winNew.center[0])), int(round(winNew.center[1])))
    winNew.size = (int(round(winNew.size[0])), int(round(winNew.size[1])))
    return  winNew

def judgment(win, rect, lost):
    minmaxX = min(win.center[0]+win.size[0]/2.0, rect[0]+rect[2])
    maxminX = max(win.center[0]-win.size[0]/2.0, rect[0])
    minmaxY = min(win.center[1]+win.size[1]/2.0, rect[1]+rect[3])
    maxminY = max(win.center[1]-win.size[1]/2.0, rect[1])
    if (minmaxX>maxminX and minmaxY>maxminY):
        ratio = (minmaxX-maxminX)*(minmaxY-maxminY)*1.0/(win.size[0]*win.size[1]+rect[2]*rect[3])
    else:
        ratio = 0
    if ratio<0.3:
        win.center = (rect[0]+rect[2]/2, rect[1]+rect[3]/2)
        win.size = (rect[2], rect[3])
        win.angle = 0
        lost += 1
    return win, lost

def update(templates, weights, imgName, imgNum, win):
    img = cv2.imread(imgName, 1)
    affMat = cv2.getRotationMatrix2D(tuple(win.center), win.angle, 1.0)
    rotatedSrc = cv2.warpAffine(img, affMat, img.shape[1:3])
    imcrop = cv2.getRectSubPix(rotatedSrc, tuple(win.size), tuple(win.center))
    imcrop = cv2.resize(imcrop,(64,64))
    #imcrop = np.reshape(imcrop, (1,)+imcrop.shape)
    template = np.zeros((3,64,64), dtype = int)
    for k in range(3):
        template[k] = imcrop[:,:,k]
    templates.pop(-1)
    templates.insert(1,template)
 
def draw(videoWri, imgName, imgNum, win, lost):
    R = np.array([[math.cos(win.angle*PI/180), -math.sin(win.angle*PI/180)], [math.sin(win.angle*PI/180), math.cos(win.angle*PI/180)]])
    P = np.array([[-win.size[0]/2.0, win.size[0]/2.0, win.size[0]/2.0, -win.size[0]/2.0], [-win.size[1]/2.0, -win.size[1]/2.0, win.size[1]/2.0, win.size[1]/2.0]])
    P = np.dot(R, P)
    P = P + np.tile(win.center, (4,1)).transpose()
    P = P[:,[0,1,2,3,0]]
    P = np.round(P).astype(int)
    im = cv2.imread(imgName)
    for k in range(5):
        cv2.line(im, tuple(P[:,k]), tuple(P[:,(k+1)%4]), cv2.cv.CV_RGB(255,0,0), 3)
    title = 'frame {:d}, lost={:d}'.format(imgNum, lost)
    cv2.putText(im, title, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.cv.CV_RGB(255,0,0), 4)
    cv2.imshow('im',im)
    cv2.waitKey(1)
    videoWri.write(im)


       
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s sequence_name'.format(sys.argv[0])
    else:
        seqName = sys.argv[1]
        sequence = os.path.join(dataSet, seqName)
        rin = open(sequence + '/groundtruth_rect.txt', 'r')
        templates = []
        weights = np.zeros((templateSize*sampleSize), dtype = float)/(templateSize*sampleSize)
        weights[0] = 1
        win = winParam((0,0), (0,0), 0)
        template = np.zeros((3,64,64), dtype = int)
        modelFile = 'train_val_affine_deploy.prototxt'
        pretrainedFile = 'affine_iter_20000.caffemodel.backup'
        net = caffe.Net(modelFile, pretrainedFile)
        net.set_phase_test()
        net.set_mode_gpu()
        lost = 0
        for imgNum in range(1, 2000):
            line = rin.readline()
            if line == 'none':
                break
            rect = np.array(line.split(' '), dtype = int)
            if imgNum == 1:
                win = winParam((rect[0]+rect[2]/2, rect[1]+rect[3]/2), (rect[2], rect[3]), 0)
                img = cv2.imread(sequence + '/img/{:04d}.jpg'.format(imgNum), 1)
                imcrop= img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                imcrop = cv2.resize(imcrop, (64,64))
                for k in range(3):
                    template[k] = imcrop[:,:,k]
                for j in range(templateSize):
                    templates.append(template)
                videoWri = cv2.VideoWriter(seqName+'.avi', cv2.cv.CV_FOURCC('F','M','P','4'), 20, (img.shape[1], img.shape[0]))
                if(not videoWri.isOpened()):
                    print 'Unable to write the video'
            else:
                imgName = sequence + '/img/{:04d}.jpg'.format(imgNum)
                winNew = prediction(net, imgName, win, templates, weights)
                win, lost = judgment(winNew, rect, lost)
                update(templates, weights, imgName, imgNum, win)
                draw(videoWri, imgName, imgNum, win, lost)
        #videoWri.release()
        #cv2.destroyAllWindows()

