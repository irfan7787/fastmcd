import numpy as np
import cv2
import MCDWrapper
from dataReader import dataReader




np.set_printoptions(precision=2, suppress=True)
mcd = MCDWrapper.MCDWrapper()
isFirst = True

# main path of data
# path = '/home/gentoowork/Desktop/dataset/PTZ/zoomInZoomOut/'
path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/zoomInZoomOut/'
# path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/twoPositionPTZCam/'


# counter for mask files
counter = 1

# reading input and groundtruth files
dr = dataReader()
files = dr.readFiles(str(path)+'input', ".jpg")
groundTruthFiles = dr.readFiles(str(path)+'groundtruth', ".png")

#roiMask = cv2.cvtColor(roiMask, cv2.COLOR_RGB2GRAY)


#tempROIData = open(path+temporalRoi, 'r')
#tempRoiArray = tempROIData.read().split(' ')

isSaveMask = True


for f in files:
    frame = cv2.imread(f)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if isFirst:
        mcd.init(gray)
        isFirst = False
    else:
        mask = mcd.run(gray)
    frame[mask > 0, 2] = 255
    if isSaveMask:
        tempName = f.split("/")[-1].replace("in","mask")
        cv2.imwrite(str(path)+'mcdMask/'+tempName, mask)
        counter = counter+1
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



