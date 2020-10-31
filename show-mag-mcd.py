import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataReader import dataReader
from utils import flow2motion

_temp = __import__("flownet2-pytorch.utils.flow_utils", globals(), locals(), ['flow2img', 'readFlow'], 0)
flow2img = _temp.flow2img
readFlow = _temp.readFlow



path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/intermittentPan/'

dr = dataReader()
files = dr.readFiles(str(path)+'input', ".jpg")
groundTruthFiles = dr.readFiles(str(path)+'groundtruth', ".png")
flowFiles = dr.readFiles(path+'flow', 'flo')

mcdResults = dr.readFiles(str(path)+'mcdMask', ".jpg")
mcdFlowResults = dr.readFiles(path+'mcdFlowMask', 'jpg')
flowFiles = dr.readFiles(path+'flow', 'flo') 


idx = 1623  #1830  #851

img = cv2.imread(files[idx])
height, width, _ = img.shape

gt = cv2.imread(groundTruthFiles[idx])

mcd = cv2.imread(mcdResults[idx], 0)
mcdFlow = cv2.imread(mcdFlowResults[idx])

flow = readFlow(flowFiles[idx-8])
flowImg = flow2img(flow)
mag, motion = flow2motion(flow)


out = mcd
changeBgMean = np.mean(mag[out==0])
changeFgMean = np.mean(mag[out>0])

print("backgroung mean change of magnitude: %.3f  FG: %.3f" %(changeBgMean, changeFgMean))
meanMotionFlow = np.mean(mag[motion>0])
print("mean of motion detected from Flow: ", meanMotionFlow)

cv2.imshow("img", img)
cv2.imshow("gt", gt)
cv2.imshow("mcd", mcd)
cv2.imshow("mag", mcdFlow)
cv2.imshow("motion from flow", flowImg)


fig, axarr = plt.subplots(1,6)

axarr[0].imshow(img)
axarr[0].axis('off')
axarr[0].set_title('Input')

axarr[1].imshow(gt, cmap='gray')
axarr[1].axis('off')
axarr[1].set_title('Ground truth')

axarr[2].imshow(mcd, cmap='gray')
axarr[2].axis('off')
axarr[2].set_title('fastMCD')

axarr[3].imshow(mag, cmap='gray')
axarr[3].axis('off')
axarr[3].set_title('Flow magnitude')

axarr[4].imshow(motion, cmap='gray')
axarr[4].axis('off')
axarr[4].set_title('Magnitude Otsu threshold')

if (meanMotionFlow - changeBgMean) < 10:
    print("flow motion result can not be trusted!!")
    motion[:] = 0

axarr[5].imshow(motion, cmap='gray')
axarr[5].axis('off')
axarr[5].set_title('M(flow)')

plt.show()


cv2.waitKey(0)

