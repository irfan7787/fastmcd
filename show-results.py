import matplotlib.pyplot as plt
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


idx = 1830  #851

img = cv2.imread(files[idx])
height, width, _ = img.shape

gt = cv2.imread(groundTruthFiles[idx])

mcd = cv2.imread(mcdResults[idx])
mcdFlow = cv2.imread(mcdFlowResults[idx])

flow = readFlow(flowFiles[idx-8])
flowImg = flow2img(flow)

flowImg[0,:] = 0
flowImg[height-1,:] = 0
flowImg[:,0] = 0
flowImg[:,width-1] = 0

cv2.imshow("img", img)
cv2.imshow("gt", gt)
cv2.imshow("mcd", mcd)
cv2.imshow("mcd-Flow", mcdFlow)
cv2.imshow("flownetCSS", flowImg)


fig, axarr = plt.subplots(1,5)

axarr[0].imshow(img)
axarr[0].axis('off')
axarr[0].set_title('Input')

axarr[1].imshow(gt)
axarr[1].axis('off')
axarr[1].set_title('Ground truth')

axarr[2].imshow(flowImg)
axarr[2].axis('off')
axarr[2].set_title('Flow')

axarr[3].imshow(mcd)
axarr[3].axis('off')
axarr[3].set_title('fastMCD result')

axarr[4].imshow(mcdFlow)
axarr[4].axis('off')
axarr[4].set_title('fastMCD + flow')


plt.show()


cv2.waitKey(0)

