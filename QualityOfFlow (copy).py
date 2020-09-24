from dataReader import dataReader
from qualityAnalysis import  qualityAnalysis
import cv2
import numpy as np
from numpy import linalg as LA
import getpass

_temp = __import__("flownet2-pytorch.utils.flow_utils", globals(), locals(), ['flow2img', 'readFlow'], 0)
flow2img = _temp.flow2img
readFlow = _temp.readFlow


def findMag(du, dv):
    max_flow = np.abs(max(np.max(du), np.max(dv)))
    mag = np.sqrt(du * du + dv * dv) * 8 / max_flow 
    return mag

def magToImg(mag):
    mag = np.float32(mag)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# main path of data
path = '../Desktop/dataset/PTZ/zoomInZoomOut/'
if getpass.getuser() == 'ibrahim':
    path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/twoPositionPTZCam/'

roiName = 'ROI.bmp'

f = open(path+"temporalROI.txt", "r")
line = f.readline()
temporalRoiFirst, temporalRoiLast = [int(i) for i in line.split()]
f.close()

# reading input and groundtruth files
dr = dataReader()
imageFiles = dr.readFiles(str(path)+'input', 'jpg')
groundTruthFiles = dr.readFiles(str(path)+'groundtruth', ".png")
flowFiles = dr.readFiles(path+'flow', 'flo')


roi = cv2.imread(str(path)+'ROI.bmp')
binaryRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

qa = qualityAnalysis()

step = 8
imgPrev = None
isStopped = False
i = temporalRoiFirst-1
processedFrameCounter = 0
while i < temporalRoiLast:
    
    keyy = cv2.waitKey(10)
    if  keyy == ord('q'):
        break

    if keyy == ord('s'):
        isStopped = ~isStopped

    if isStopped:
        continue
    
    img = cv2.imread(imageFiles[i])
    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

    i +=1
    if groundTruth[0,0] == 170:
        continue

    print(i)
    groundTruth = np.multiply(groundTruth, binaryRoi)
    groundTruth[groundTruth>0] = 1

    flow = readFlow(flowFiles[i-step])
    flowImg = flow2img(flow)
    flowGray = cv2.cvtColor(flowImg, cv2.COLOR_BGR2GRAY)
    

    w,h,_ = flow.shape
    points = []
    for m in range(int(h//16), h, int(h//16)):
        for n in range(int(w//16), w, int(w//16)):        
            points.append([m, n])

    # imageNew = cv2.resize(image,(h,w))
    # newPoints = imageNew + flow[:,:,0] + flow[:,:,1]
    
    good1 = np.reshape(points, (len(points),2)).astype(np.float32)
    good2 = []
    for x,y in points:
        good2.append( [x + flow[y,x,0], y + flow[y,x,1]] )

    good2 = np.reshape(good2, (len(points),2))

    # self.makeHomoGraphy(good1, good2)
    homo, status = cv2.findHomography(good1, good2, cv2.RANSAC, 1.0)

    thresh = 0.1 * step + 0.3 * np.sqrt( np.power(homo[0,2],2) + np.power(homo[1,2],2) )
    print("thresh: ", thresh)


    mag = findMag(flow[:,:,0], flow[:,:,1])
    print("mag min: %.3f  max: %.3f  std: %.3f" %(np.min(mag), np.max(mag), np.std(mag)))

    mask = cv2.adaptiveThreshold(flowGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # mask2 = cv2.adaptiveThreshold(255-flowGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # mask = cv2.bitwise_or(mask, mask2)

    cv2.imshow("img", img)
    # cv2.imshow("gt", groundTruth*255)
    cv2.imshow("flow", flowImg)
    cv2.imshow("mag", mag)
    cv2.imshow("mask", mask)

    motion = mask
    motion = np.multiply(motion, binaryRoi)
    motion[motion>0] = 1
    
    qa.compare(groundTruth, motion)
    processedFrameCounter +=1
    # print(qa.printIterationResults())


print("processedFrameCounter: ", processedFrameCounter)
print("\n ***Mean values*** \n ")
print(qa.results(path))
