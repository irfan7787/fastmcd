import torch
import numpy as np
import argparse
import cv2 
import os

from models import FlowNet2  # the path is depended on where you create this module
from models import FlowNet2CSS
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils.flow_utils import flow2img
from utils.flow_utils import writeFlow

def readFiles(path, fileType):
        # reading input files
        Files = []
        for r, d, f in os.walk(path):
            for file in f:
                if fileType in file:
                    Files.append(os.path.join(r, file))

        Files.sort()

        return Files

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2CSS(args).cuda()
    dict = torch.load("/home/ibrahim/Desktop/ornek-projeler/flownet2-pytorch/checkpoints/FlowNet2-CSS-ft-sd_checkpoint.pth.tar")
    # net = FlowNet2(args).cuda()
    # dict = torch.load("/home/ibrahim/Desktop/ornek-projeler/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar")
    
    net.load_state_dict(dict["state_dict"])

    path = '/home/ibrahim/Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/continuousPan/'

    inputFiles = readFiles(path+'input', 'jpg')
    step = 1
    isStopped = False
    i = 0
    while i<len(inputFiles):

        keyy = cv2.waitKey(10)
        if  keyy == ord('q'):
            break

        if keyy == ord('s'):
            isStopped = ~isStopped

        if isStopped:
            continue
        
        # load the image pair, you can find this operation in dataset.py
        pim1 = read_gen(inputFiles[i])
        pim2 = read_gen(inputFiles[i-step])
        i +=1

        height, width, d = pim1.shape
        isResized = False

        if width == 320 and height == 240:
            pim1 = cv2.resize(pim1, (128,64))
            pim2 = cv2.resize(pim2, (128,64))
            isResized = True
        elif width == 704 and height == 480:
            pim1 = cv2.resize(pim1, (192,128))
            pim2 = cv2.resize(pim2, (192,128))
            isResized = True
        elif width == 560 and height == 368:
            pim1 = cv2.resize(pim1, (192,128))
            pim2 = cv2.resize(pim2, (192,128))
            isResized = True
        elif width == 570 and height == 340:
            pim1 = cv2.resize(pim1, (192,128))
            pim2 = cv2.resize(pim2, (192,128))
            isResized = True

        start = cv2.getTickCount()
        images = [pim1, pim2]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # process the image pair to obtian the flow
        result = net(im).squeeze()

        data = result.data.cpu().numpy().transpose(1, 2, 0)
        # data = (data * 100).astype(np.uint8)
        if isResized:   
            width =  (width//4) 
            height =  (height//4)
            data = cv2.resize(data, (width, height))

        elapsed_time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
        print ('**************** elapsed time: %.3fs'%elapsed_time)

        tempName = inputFiles[i].replace("input","flow-grid")
        tempName = tempName.replace(".jpg",".flo")
        writeFlow(tempName, data)

        cv2.imshow("img",pim1)
        cv2.imshow("flow",flow2img(data))
        cv2.waitKey(10)
        
