import numpy as np
from sklearn.metrics import confusion_matrix

# sample matrices
# groundTruth = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 0]])
# predicted = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0]])

class qualityAnalysis:

    def __init__(self):

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.iouList = []
        self.precisionList = []
        self.recallList = []
        self.f1List = []
        self.diceList = []
        self.fprList = []
        self.fnrList = []
        self.pwcList = []
        self.specificityList = []



    def __IoU(self, groundTruth, predicted):
        # intersection over uniun (IoU)
        intersection = np.sum(np.multiply(groundTruth, predicted))
        union = np.sum((groundTruth + predicted) > 0)
        IoU = intersection / union
        self.iouList.append(IoU)
        return IoU

    # def confusionMatrix(self):
    #
    #     return [[self.TP, self.FP], [self.FN, self.TN]]

    def __dice(self):
        dice = 2 * self.TP / (self.FP + 2 * self.TP + self.FN)
        self.diceList.append(dice)
        return dice

    def __recall(self):
        recall = self.TP / (self.TP + self.FN)
        self.recallList.append(recall)
        return recall

    def __precision(self):
        precision = self.TP / (self.TP + self.FP)
        self.precisionList.append(precision)

    def __F1(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        f1 = (2 * precision * recall) / (precision + recall)
        self.f1List.append(f1)
        return f1

    def __PWC(self):
        pwc = 100 * (self.FN + self.FP) / (self.TP + self.TN + self.FP + self.FN)  # percentage of wrong classification
        self.pwcList.append(pwc)
        return pwc

    def __FPR(self):
        fpr = self.FP / (self.FP + self.TN)  # false positive rate
        self.fprList.append(fpr)
        return fpr

    def __FNR(self):
        fnr = self.FN / (self.TP + self.FN)  # false negative rate
        self.fnrList.append(fnr)
        return fnr

    def __specificity(self):

        spec = self.TN / (self.TN + self.FP)
        self.specificityList.append(spec)
        return spec

    def find_metrics(self, true_values, estimated):
    
        confusion = confusion_matrix(true_values.ravel(),estimated.ravel())
        
        tn = confusion[0,0]
        fn = confusion[1,0]
        tp = confusion[1,1]
        fp = confusion[0,1]
        
        return tn, fn, tp, fp

    def compare(self, groundTruth, predicted):
        groundTruth = groundTruth
        predicted = predicted

        # confusion matrix
        self.TN, self.FN, self.TP, self.FP = self.find_metrics(groundTruth, predicted)

        self.__dice()
        self.__F1()
        self.__FNR()
        self.__FPR()
        self.__IoU(groundTruth, predicted)
        self.__precision()
        self.__recall()
        self.__PWC()
        self.__specificity()

    def printIterationResults(self):

        return [self.diceList[-1], self.f1List[-1], self.fnrList[-1], self.fprList[-1], self.iouList[-1], self.precisionList[-1], self.recallList[-1], self.specificityList[-1]]

    def results(self):

        dice = np.mean(self.diceList)
        f1 = np.mean(self.f1List)
        fnr = np.mean(self.fnrList)
        fpr = np.mean(self.fprList)
        iou = np.mean(self.iouList)
        precision = np.mean(self.precisionList)
        recall = np.mean(self.recallList)
        spec = np.mean(self.specificityList)

        return [dice, f1, fnr, fpr, iou, precision, recall, spec]




