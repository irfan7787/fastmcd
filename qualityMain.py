from dataReader import dataReader
from qualityAnalysis import qualityAnalysis
import cv2
import numpy as np
import getpass

# main path of data

if getpass.getuser() == 'gentoowork':
    path = '/home/gentoowork/Desktop/motionDetectionData/PTZ/continuousPan/'

if getpass.getuser() == 'gentoo':
    path = '/home/gentoo/Desktop/motionDetectionData/PTZ/twoPositionPTZCam/'

if getpass.getuser() == 'ibrahim':
    path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/zoomInZoomOut/'

roiName = 'ROI.bmp'
temporalRoiFirst = 800
temporalRoiLast = 2300

# reading input and groundtruth files
dr = dataReader()
groundTruthFiles = dr.readFiles(str(path) + 'groundtruth', ".png")
maskFiles = dr.readFiles(str(path) + 'mcdMask', 'jpg')
inputFiles = dr.readFiles(str(path) + 'input', 'jpg')

roi = cv2.imread(str(path) + 'ROI.bmp')
binaryRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# binaryRoi = abs(255-binaryRoi)
binaryRoi[binaryRoi > 0] = 1
cv2.imshow('binary', binaryRoi * 255)
qa = qualityAnalysis()

for i in range(temporalRoiFirst - 1, temporalRoiLast):
    # for i in range(909, 912):
    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    groundTruth[groundTruth > 0] = 1
    mask = cv2.imread(maskFiles[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 1
    input = cv2.imread(inputFiles[i])

    # cv2.imshow('input', input)
    # cv2.imshow('mcd', mask)
    # cv2.imshow('groundtruth', groundTruth)

    if (groundTruth.size != np.sum(groundTruth)):  # skipping non-labeled frames
        groundTruth = np.multiply(groundTruth, binaryRoi)
        mask = np.multiply(mask, binaryRoi)
        qa.compare(groundTruth, mask)
        print(i)
        print(qa.printIterationResults())
    #cv2.waitKey(0)

print("\n ***Mean values*** \n ")
print(qa.results(path))
