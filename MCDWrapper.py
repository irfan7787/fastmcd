import numpy as np
import cv2
import KLTWrapper
import ProbModel


class MCDWrapper:
    def __init__(self):
        self.imgIpl = None
        self.imgGray = None
        self.imgGrayPrev = None
        self.frm_cnt = 0
        self.lucasKanade = KLTWrapper.KLTWrapper()
        self.model = ProbModel.ProbModel()

    def init(self, image):
        self.imgGray = image
        self.imgGrayPrev = image
        self.lucasKanade.init(self.imgGray)
        self.model.init(self.imgGray)

    def setApplyFlow(self):
        self.model.setApplyFlow()

    def run(self, frame, flow):
        self.frm_cnt += 1
        self.imgIpl = frame
        self.imgGray = frame
        # self.imgGray = cv2.medianBlur(imgGray, 5)
        if self.imgGrayPrev is None:
            self.imgGrayPrev = self.imgGray.copy()
        
        self.lucasKanade.RunTrack(self.imgGray, self.imgGrayPrev)
        # self.lucasKanade.RunTrackwithFlow(self.imgGray, flow)
        self.model.motionCompensate(self.lucasKanade.H, flow)
        mask = self.model.update(frame, flow)
        self.imgGrayPrev = self.imgGray.copy()
        return mask





