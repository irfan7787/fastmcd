import numpy as np
import cv2
import MCDWrapper
from dataReader import dataReader
import getpass

# from flownet2-pytorch.utils.flow_utils import flow2img
_temp = __import__("flownet2-pytorch.utils.flow_utils", globals(), locals(), ['flow2img', 'readFlow'], 0)
flow2img = _temp.flow2img
readFlow = _temp.readFlow


def findMagAngle(du, dv):
    # max_flow = np.abs(max(np.max(du), np.max(dv)))
    # mag = np.sqrt(du * du + dv * dv) * 8 / max_flow 
    mag, angle = cv2.cartToPolar(du, dv)
    return mag, angle

def magToImg(mag):
    mag = np.float32(mag)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

np.set_printoptions(precision=2, suppress=True)
mcd = MCDWrapper.MCDWrapper()
isFirst = True

# main path of data
path = '/home/gentoo/Desktop/motionDetectionData/PTZ/zoomInZoomOut/'

if getpass.getuser() == 'ibrahim':
    path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/continuousPan/'
    # path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/zoomInZoomOut/'


# counter for mask files
counter = 1

# reading input and groundtruth files
dr = dataReader()
files = dr.readFiles(str(path)+'input', ".jpg")
groundTruthFiles = dr.readFiles(str(path)+'groundtruth', ".png")
flowFiles = dr.readFiles(path+'flow', 'flo')

#roiMask = cv2.cvtColor(roiMask, cv2.COLOR_RGB2GRAY)


#tempROIData = open(path+temporalRoi, 'r')
#tempRoiArray = tempROIData.read().split(' ')

applyFlow = True
isSaveMask = True

savingFolder = "mcdMask/"
if applyFlow:
    savingFolder = "mcdFlowMask/"

isStopped = False
i = 0
while i<len(files):

    keyy = cv2.waitKey(10)
    if  keyy == ord('q'):
        break

    if keyy == ord('s'):
        isStopped = ~isStopped

    if isStopped:
        continue

    if i >=len(flowFiles):
        break

    # print(i)

    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

    f = files[i]
    frame = cv2.imread(f)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    flow = readFlow(flowFiles[i-8])

    i +=1

    flowImg = flow2img(flow)
    flowImgGray = cv2.cvtColor(flowImg, cv2.COLOR_BGR2GRAY)

    cv2.imshow("flow", flowImg)

    height, width = gray.shape
    isResized = False

    if width%4 !=0 or height%4!=0:
        gray = cv2.resize(gray, ( 4 * (width//4), 4 * (height//4)))
        isResized = True

    mask = np.zeros(gray.shape, np.uint8)
    if isFirst:
        mcd.init(gray)
        isFirst = False
    else:
        mask = mcd.run(gray, flow)

    if isResized:
        mask = cv2.resize(mask, (width, height))

    frame[mask == 255, 2] = 255
    frame[mask == 100, 0] = 255

    if isSaveMask:
        tempName = f.split("/")[-1].replace("in", "mask")
        cv2.imwrite(str(path)+ savingFolder + tempName, mask)
        counter = counter+1
    cv2.imshow('frame', frame)

