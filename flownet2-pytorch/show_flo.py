import os
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

def readFiles(path, fileType):
        # reading input files
        Files = []
        for r, d, f in os.walk(path):
            for file in f:
                if fileType in file:
                    Files.append(os.path.join(r, file))

        Files.sort()

        return Files

def findMag(du, dv):
    max_flow = np.abs(max(np.max(du), np.max(dv)))
    mag = np.sqrt(du * du + dv * dv) * 8 / max_flow 
    return mag

def magToImg(mag):
    mag = np.float32(mag)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


path = 'dataset/PTZ/zoomInZoomOut/'

if getpass.getuser() == 'ibrahim':
    path = '/home/ibrahim/Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/zoomInZoomOut/'

inputFiles = readFiles(path+'input', 'jpg')
flowFiles = readFiles(path+'flow', 'flo')

for i in range(0, len(inputFiles)-1):
    img = cv2.imread(inputFiles[i])
    flow = readFlo(flowFiles[i+1], img.shape[1], img.shape[0])

    cv2.imshow("img", img)

    mag = findMag(flow[:,:,0], flow[:,:,1])
    magIm = magToImg(mag)
    cv2.imshow("flow", magIm)
    cv2.waitKey(0)
