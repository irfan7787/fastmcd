import cv2
import numpy as np


_temp = __import__("flownet2-pytorch.utils.flow_utils", globals(), locals(), ['flow2img'], 0)
flow2img = _temp.flow2img

def flow2motion(flow):

    #flowImg = flow2img(flow)
    # flowGray = cv2.cvtColor(flowImg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("flowGray", flowGray)

    mag, _ = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    magIm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("mag", magIm)
    # print("mag min: %.3f  max: %.3f  std: %.3f" %(np.min(mag), np.max(mag), np.std(mag)))

    # motion = cv2.adaptiveThreshold(magIm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    ret, motion = cv2.threshold(magIm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return mag, motion





    # mask = cv2.adaptiveThreshold(magIm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # # mask2 = cv2.adaptiveThreshold(255-flowGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    # # mask = cv2.bitwise_or(mask, mask2)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bboxes = []
    # regions = []
    # for cnt in (contours):
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
    #     # if regionMag - meanDense < stdMagDense:
    #         #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #         pass
    #     else:
    #         regions.append(cnt)
    #         bboxes.append([x,y,w,h])


    # motion = np.zeros((mask.shape))
    # cv2.drawContours(motion, regions, -1, (255), -1) 