from dataReader import dataReader
from qualityAnalysis import  qualityAnalysis
import cv2
import numpy as np
import getpass

# main path of data
path = '/home/gentoo/Desktop/motionDetectionData/PTZ/intermittentPan/'
if getpass.getuser() == 'ibrahim':
    path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/zoomInZoomOut/'

roiName = 'ROI.bmp'
temporalRoiFirst = 1200
temporalRoiLast = 1600

# reading input and groundtruth files
dr = dataReader()
groundTruthFiles = dr.readFiles(str(path) + 'groundtruth', ".png")
maskFiles = dr.readFiles(str(path) + 'mcdMask', 'jpg')

roi = cv2.imread(str(path) + 'ROI.bmp')
binaryRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
binaryRoi[binaryRoi > 0] = 1
qa = qualityAnalysis()

for i in range(temporalRoiFirst - 1, temporalRoiLast):
    # for i in range(1406, 1409):
    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    groundTruth[groundTruth > 0] = 1
    mask = cv2.imread(maskFiles[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1
    if (groundTruth.size != np.sum(groundTruth)):  # skipping non-labeled frames
        groundTruth = np.multiply(groundTruth, binaryRoi)
        mask = np.multiply(mask, binaryRoi)
        qa.compare(groundTruth, mask)
        print(i)
        print(qa.printIterationResults())

print("\n ***Mean values*** \n ")
print(qa.results())
