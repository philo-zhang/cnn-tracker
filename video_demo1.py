import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import IncrementalPCA
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
sampleSize = 30
labelSize = 8
PI = 3.1415926

dataSet = '/media/philo/1T_HardDisk/'
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

def show_img(candidates):
    for i in range(sampleSize):
        for j in range(templateSize):
            candImg = np.zeros((64,64), dtype = float)
            candidate = np.reshape(candidates[i*templateSize+j],(64,64))
            plt.subplot(sampleSize, templateSize, i*templateSize+j+1)
            plt.imshow(candidate, cmap = cm.Greys_r, vmin = np.amin(candidate), vmax = np.amax(candidate))
            plt.title('{:d}'.format(i*templateSize+j+1))
            plt.axis('off')
    plt.show()

def sampling(win, sampleSize):
    xSamples = np.rint(np.random.normal(win.center[0], 10, sampleSize)).astype(int)
    ySamples = np.rint(np.random.normal(win.center[1], 10, sampleSize)).astype(int)
    angleSamples = np.random.normal(win.angle, 1, sampleSize)
    widthSamples = np.rint(np.random.normal(win.size[0], 3, sampleSize)).astype(int)
    heightSamples = np.rint(np.random.normal(win.size[1], 3, sampleSize)).astype(int)
    winSamples = []
    winWeights = []
    for i in range(sampleSize):
        winSample = winParam((0,0), 0, 0)
        winSample.center = (xSamples[i], ySamples[i])
        winSample.angle = angleSamples[i]
        winSample.size = (widthSamples[i], heightSamples[i])
        winSamples.append(winSample)
        #draw(videoWri, imgName, imgNum, winSample, lost)
        #cv2.waitKey(0)
    #for i in range(sampleSize):
        #print 'winsamples:', winSamples[i].center
    return winSamples

def maxLikelihood(img, ipca, winsNew):
    error = []
    imcropLin = np.zeros((templateSize*sampleSize, 64*64), dtype = float)
    for i in range(templateSize*sampleSize):
        win = winsNew[i]
        affMat = cv2.getRotationMatrix2D(tuple(win.center), win.angle, 1.0)
        rotatedSrc = cv2.warpAffine(img, affMat, (img.shape[1],img.shape[0]))
        imcrop = cv2.getRectSubPix(rotatedSrc, tuple(win.size), tuple(win.center))
        #cv2.imshow('rotated', rotatedSrc)
        #cv2.imshow('crop',imcrop)
        #cv2.waitKey(0)
        imcrop = cv2.resize(imcrop,(64,64))
        imcropGray = 0.114*imcrop[:,:,0]+0.587*imcrop[:,:,1]+0.299*imcrop[:,:,2]
        imcropLin[i] = np.reshape(imcropGray, (1, 64*64))
        #show_img(imcropLin[i])
        #cv2.waitKey(0)
    imcropLin = imcropLin-ipca.mean_
    if ipca.components_ == []:
        score = np.sum(imcropLin**2, axis=1)
    else:
        eigensSqr = np.dot(ipca.components_.transpose(), ipca.components_)
        print eigensSqr.shape, imcropLin.shape
        diff = imcropLin-np.dot(imcropLin, eigensSqr)
        score = np.sum(diff**2, axis=1)
    ind = np.argmin(score)
    #print 'score', score
    print 'ind', ind
    return winsNew[ind]


    
def prediction(net, img, win, templates, weights, ipca):
    winSamples = sampling(win, sampleSize)
    #winSamples = []
    #winSamples.append(win)
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
        for j in range(3):
            candidate[j] =imcrop[:,:,j]
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

    templates = (templates-XmeanM)/255.0
    candidates = (candidates-XmeanM)/255.0
    #print 'diff between templates[0] and [1]:', templates[0]-templates[1]
    X = np.concatenate((templates, candidates), 1)
    print 'X.shape=', X.shape
    out = net.forward_all(data=X)
    output = out['fc7']
    output = output.reshape(*(output.shape[0:2]))
    ratios = np.tile(ratios, (1,4))
    verticesNew = vertices - np.multiply(output, ratios)
    #print vertices, verticesNew
    vertex = np.array([win.center[0]-win.size[0]/2, win.center[1]-win.size[1]/2, win.center[0]+win.size[0]/2, win.center[1]-win.size[1]/2, win.center[0]+win.size[0]/2, win.center[1]+win.size[1]/2, win.center[0]-win.size[0]/2, win.center[1]+win.size[1]/2])

    dCenter, dSize, dAngle, transM = pose_estimation(vertices, verticesNew)
    #print 'output:', output
    #print 'verticesNew:', verticesNew
    #print 'transM:', transM
    #print 'dcenter:', dCenter
    #print 'dSize:', dSize
    #print 'dcenterApp:', np.rint(dCenter).astype(int)
    #print 'dAngle:', dAngle
    #print weights
    winsNew = []
    for i in range(sampleSize):
        for j in range(templateSize):
            winNew = winParam((0,0),(0,0),0)
            winNew.center = winSamples[i].center + np.rint(dCenter[i*templateSize+j]).astype(int)
            winNew.size = np.rint(winSamples[i].size*dSize[i*templateSize+j]).astype(int)
            winNew.angle = winSamples[i].angle + dAngle[i*templateSize+j]
            #draw(videoWri, imgName, imgNum, winNew, lost)
            #cv2.waitKey(0)
            winsNew.append(winNew)
            #print j, winsNew[j].center
    #print 'winsNew[0]:', winsNew[0].center, winsNew[0].size
    #print 'winsNew[1]:', winsNew[1].center, winsNew[1].size
    winMax = maxLikelihood(img, ipca, winsNew)
    print 'winMax:',winMax.center, winMax.size, winMax.angle
    print 'win:', win.center, win.size, win.angle
    return  winMax

def judgment(win, rect, lost):
    minmaxX = min(win.center[0]+win.size[0]/2.0, rect[0]+rect[2])
    maxminX = max(win.center[0]-win.size[0]/2.0, rect[0])
    minmaxY = min(win.center[1]+win.size[1]/2.0, rect[1]+rect[3])
    maxminY = max(win.center[1]-win.size[1]/2.0, rect[1])
    if (minmaxX>maxminX and minmaxY>maxminY):
        ratio = (minmaxX-maxminX)*(minmaxY-maxminY)*1.0/(win.size[0]*win.size[1]+rect[2]*rect[3])
    else:
        ratio = 0
    if ratio<0.1:
        win.center = (rect[0]+rect[2]/2, rect[1]+rect[3]/2)
        win.size = (rect[2], rect[3])
        win.angle = 0
        lost += 1
    return win, lost

def update(templates, img, winMax, ipca):
    affMat = cv2.getRotationMatrix2D(tuple(winMax.center), winMax.angle, 1.0)
    rotatedSrc = cv2.warpAffine(img, affMat, (img.shape[1], img.shape[0]))
    imcrop = cv2.getRectSubPix(rotatedSrc, tuple(winMax.size), tuple(winMax.center))
    imcrop = cv2.resize(imcrop,(64,64))
    #cv2.imshow('imcrop',imcrop)
    #cv2.waitKey(0)
    #imcrop = np.reshape(imcrop, (1,)+imcrop.shape)
    template = np.zeros((3,64,64), dtype = int)
    for k in range(3):
        template[k] = imcrop[:,:,k]
    templates.pop(-1)
    templates.insert(1,template)
    #print templates[0]-templates[1]
    templateGray = 0.114*template[0,:,:]+0.587*template[1,:,:]+0.299*template[2,:,:]
    print templateGray.shape
    templateGray = np.reshape(templateGray, (1, 64*64))
    ipca.partial_fit(templateGray)
    return templates, ipca
 
def draw(videoWri, img, imgNum, win, lost):
    R = np.array([[math.cos(win.angle*PI/180), -math.sin(win.angle*PI/180)], [math.sin(win.angle*PI/180), math.cos(win.angle*PI/180)]])
    P = np.array([[-win.size[0]/2.0, win.size[0]/2.0, win.size[0]/2.0, -win.size[0]/2.0], [-win.size[1]/2.0, -win.size[1]/2.0, win.size[1]/2.0, win.size[1]/2.0]])
    P = np.dot(R, P)
    P = P + np.tile(win.center, (4,1)).transpose()
    P = P[:,[0,1,2,3,0]]
    P = np.rint(P).astype(int)
    for k in range(5):
        cv2.line(img, tuple(P[:,k]), tuple(P[:,(k+1)%4]), cv2.cv.CV_RGB(255,0,0), 3)
    title = 'frame {:d}, lost={:d}'.format(imgNum, lost)
    cv2.putText(img, title, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.cv.CV_RGB(255,0,0), 4)
    cv2.imshow('im',img)
    cv2.waitKey(10)
    videoWri.write(img)


       
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s sequence_name'.format(sys.argv[0])
    else:
        seqName = sys.argv[1]
        sequence = os.path.join(dataSet, seqName)
        templates = []
        weights = np.zeros((templateSize*sampleSize), dtype = float)/(templateSize*sampleSize)
        weights[0] = 1
        win = winParam((0,0), (0,0), 0)
        modelFile = 'train_val_affine_deploy.prototxt'
        pretrainedFile = 'affine_iter_20000.caffemodel.backup'
        net = caffe.Net(modelFile, pretrainedFile)
        net.set_phase_test()
        net.set_mode_gpu()
        lost = 0
        pca_mean = np.zeros((1, 64*64*3), dtype = int)
        pca_eigens = []
        rect = [210, 50, 90, 240] 
        cap = cv2.VideoCapture(sequence)
        if not cap.isOpened():
            print "cannot open the video"
        for i in range(10):
            ret, img = cap.read()
        for imgNum in range(1, 2000):
            if imgNum == 1:
                template = np.zeros((3,64,64), dtype = int)
                win = winParam((rect[0]+rect[2]/2, rect[1]+rect[3]/2), (rect[2], rect[3]), 0)
                ret, img = cap.read()
                img = np.array(img)
                imcrop= img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                imcrop = cv2.resize(imcrop, (64,64))
                for k in range(3):
                    template[k] = imcrop[:,:,k]
                templateGray = 0.114*template[0]+0.587*template[1]+0.299*template[2]
                for j in range(templateSize):
                    templates.append(template)
                pca_mean = np.reshape(templateGray, (1,64*64))
                pca_data = np.tile(pca_mean, (templateSize,1))
                print pca_data.shape
                ipca = IncrementalPCA(n_components=10, whiten=1)
                ipca.fit(pca_data)
                print ipca.components_.shape, ipca.mean_.shape
                videoWri = cv2.VideoWriter(seqName+'.avi', cv2.cv.CV_FOURCC('F','M','P','4'), 20, (img.shape[1], img.shape[0]))
                if(not videoWri.isOpened()):
                    print 'Unable to write the video'
            else:
                ret, img = cap.read()
                img = np.array(img)
                win = prediction(net, img, win, templates, weights, ipca)
                #win, lost = judgment(winMax, rect, lost)
                templates, ipca = update(templates, img, win, ipca)
                draw(videoWri, img, imgNum, win, lost)
        #videoWri.release()
        #cv2.destroyAllWindows()

