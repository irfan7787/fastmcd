from dataReader import dataReader
from qualityAnalysis import  qualityAnalysis
import cv2
import numpy as np
import getpass

def readFlo(filename, w, h):
    f = open(filename,'rb')
    data = np.fromfile(f, np.int8, count=2*w*h) / 100  # we have stored with x 100 while saving .flo
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()
    return flow


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
    path = '../Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/intermittentPan/'

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

for i in range(temporalRoiFirst-1, temporalRoiLast):
# for i in range(651,652): 
    img = cv2.imread(imageFiles[i])

    groundTruth = cv2.imread(groundTruthFiles[i])
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

    if groundTruth[0,0] == 170:
        break
    groundTruth[groundTruth>0] = 1

    flow = readFlo(flowFiles[i], groundTruth.shape[1], groundTruth.shape[0])

    mag = findMag(flow[:,:,0], flow[:,:,1])
    magIm = magToImg(mag)
    

    mask = cv2.adaptiveThreshold(magIm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # mask2 = cv2.adaptiveThreshold(255-magIm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # mask = cv2.bitwise_or(mask1, mask2)

    cv2.imshow("img", img)
    cv2.imshow("gt", groundTruth*255)
    cv2.imshow("flow", magIm)
    cv2.imshow("mask", mask)
    stdMagDense = np.std(mag)
    

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # regions = []
    # for i, cnt in enumerate(contours):
    #     x,y,w,h = cv2.boundingRect(cnt)

    #     M = cv2.moments(cnt)
    #     if 0 == M["m00"]:
    #         continue

    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])

    #     regionMag = mag[cY,cX]
    #     # print("regionMag: ", regionMag)
        
    #     a, b = cY+ w, cX
    #     pright = np.mean(mag[a:a+5, b:b+5])

    #     a, b = cY - w, cX
    #     pleft = np.mean(mag[a-5:a, b:b+5])

    #     a, b = cY, cX - h
    #     ptop = np.mean(mag[a:a+5, b-5:b])

    #     a, b = cY, cX + h
    #     pbottom = np.mean(mag[a:a+5, b:b+5])

    #     text = str(round(pright,2))
    #     # cv2.putText(frame, text, (cX,cY), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 1, cv2.LINE_8) 
        
    #     if np.abs(regionMag - pright) < stdMagDense or np.abs(regionMag - pleft) < stdMagDense or np.abs(regionMag - ptop) < stdMagDense or np.abs(regionMag - pbottom) < stdMagDense: 
    #         #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #         pass
    #     else:
    #         regions.append(cnt)


    # motion = np.zeros((mask.shape))
    # cv2.drawContours(motion, regions, -1, (255), -1) 
    # cv2.imshow("motion", motion)

    motion = mask
    motion[motion>0] = 1
    groundTruth = np.multiply(groundTruth, binaryRoi)
    motion = np.multiply(motion, binaryRoi)
    qa.compare(groundTruth, motion)
    print(i)
    print("stdMagDense: ",stdMagDense)
    # print(qa.printIterationResults())

    if cv2.waitKey(10) == 27: 
            break  # esc to quit

print("\n ***Mean values*** \n ")
print(qa.results())
