from dataReader import dataReader
from qualityAnalysis import  qualityAnalysis
import cv2
import numpy as np

# main path of data
path = '/home/gentoowork/Desktop/dataset/PTZ/zoomInZoomOut/'

roiName = 'ROI.bmp'
temporalRoiFirst = 500
temporalRoiLast = 1130

# reading input and groundtruth files
dr = dataReader()
groundTruthFiles = dr.readFiles(str(path)+'groundtruth', ".png")
maskFiles = dr.readFiles(str(path)+'mask', 'jpg')


roi = cv2.imread(str(path)+'ROI.bmp')
binaryRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

qa = qualityAnalysis()

for i in (temporalRoiFirst-1, temporalRoiLast-1):
    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(maskFiles[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    np.multiply(groundTruth, binaryRoi)
    np.multiply(mask, binaryRoi)
    qa.newParameter(groundTruth, mask)
    print(qa.printIterationResults())

print(qa.results())
