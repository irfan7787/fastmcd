import torch
import numpy as np
import argparse
import cv2 
import os

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module


# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    # flow = flow.astype(np.float32)
    flow = (flow * 100).astype(np.int8)
    flow.tofile(f)
    f.flush()
    f.close()

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
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("/home/ibrahim/Desktop/ornek-projeler/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    path = '/home/ibrahim/Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/twoPositionPTZCam/'

    inputFiles = readFiles(path+'input', 'jpg')

    for i in range(0, len(inputFiles)-1):
        # load the image pair, you can find this operation in dataset.py
        pim1 = read_gen(inputFiles[i])
        pim2 = read_gen(inputFiles[i+1])

        height, width, d = pim1.shape
        isResized = False

        if width == 320 and height == 240:
            pim1 = cv2.resize(pim1, (320,256))
            pim2 = cv2.resize(pim2, (320,256))
            isResized = True
        elif width == 704 and height == 480:
            pim1 = cv2.resize(pim1, (704,512))
            pim2 = cv2.resize(pim2, (704,512))
            isResized = True
        elif width == 560 and height == 368:
            pim1 = cv2.resize(pim1, (576,384))
            pim2 = cv2.resize(pim2, (576,384))
            isResized = True
        elif width == 570 and height == 340:
            pim1 = cv2.resize(pim1, (576,384))
            pim2 = cv2.resize(pim2, (576,384))
            isResized = True

        start = cv2.getTickCount()
        images = [pim2, pim1]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # process the image pair to obtian the flow
        result = net(im).squeeze()

        data = result.data.cpu().numpy().transpose(1, 2, 0)
        # data = (data * 100).astype(np.uint8)
        if isResized:    
            data = cv2.resize(data, (width, height))

        elapsed_time = (cv2.getTickCount()-start)/cv2.getTickFrequency()
        print ('**************** elapsed time: %.3fs'%elapsed_time)

        tempName = inputFiles[i+1].replace("input","flow")
        tempName = tempName.replace(".jpg",".flo")
        writeFlow(tempName, data)
