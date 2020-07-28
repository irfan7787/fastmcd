import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score


# Dividing by 0

# In some rare cases, the calculation of Precision or Recall can cause a division by 0. Regarding the precision, this can happen if there are
# no results inside the answer of an annotator and, thus, the true as well as the false positives are 0. For these special cases, we have
# defined that if the true positives, false positives and false negatives are all 0, the precision, recall and F1-measure are 1. This might
# occur in cases in which the gold standard contains a document without any annotations and the annotator (correctly) returns no
# annotations. #If true positives are 0 and one of the two other counters is larger than 0, the precision, recall and F1-measure are 0.
# https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure


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
        IoU = jaccard_score(groundTruth.ravel(), predicted.ravel())
        self.iouList.append(IoU)
        return IoU

    # def confusionMatrix(self):
    #
    #     return [[self.TP, self.FP], [self.FN, self.TN]]

    def __dice(self):
        dice = 2 * self.TP / (self.FP + 2 * self.TP + self.FN)
        self.diceList.append(dice)
        return dice

    def __recall(self, groundTruth, predicted):
        # if self.TP == 0 and self.FP == 0 and self.FN == 0:
        #     recall = 1
        # elif self.TP == 0 and self.FP == 0 or self.FN == 0:
        #     recall = 0
        # else:
        #     recall = self.TP / (self.TP + self.FN)
        recall = recall_score(groundTruth.ravel(), predicted.ravel())
        self.recallList.append(recall)

    def __precision(self, groundTruth, predicted):
        # if self.TP == 0 and self.FP == 0 and self.FN == 0:
        #     precision = 1
        # elif self.TP == 0 and self.FP == 0 or self.FN == 0:
        #     precision = 0
        # else:
        #     precision = self.TP / (self.TP + self.FP)
        precision = precision_score(groundTruth.ravel(), predicted.ravel(), labels=[0, 1], zero_division=1)
        self.precisionList.append(precision)

    def __F1(self, groundTruth, predicted):
        # if self.TP == 0 and self.FP == 0 and self.FN == 0:
        #     recall = 1
        #     precision = 1
        #     f1 = 1
        # elif self.TP == 0 and self.FP == 0 or self.FN == 0:
        #     recall = 0
        #     precision = 0
        #     f1 = 0
        # else:
        #     recall = self.TP / (self.TP + self.FN)
        #     precision = self.TP / (self.TP + self.FP)
        #     if precision == 0 and recall == 0:
        #         f1 = 0
        #     else:
        #         f1 = (2 * precision * recall) / (precision + recall)
        f1 = f1_score(groundTruth.ravel(), predicted.ravel(), labels=[0, 1], zero_division=1)
        self.f1List.append(f1)

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
        confusion = confusion_matrix(true_values.ravel(), estimated.ravel(), labels=[0, 1])

        tn = confusion[0, 0]
        fn = confusion[1, 0]
        tp = confusion[1, 1]
        fp = confusion[0, 1]

        return tn, fn, tp, fp

    def compare(self, groundTruth, predicted):
        # confusion matrix
        self.TN, self.FN, self.TP, self.FP = self.find_metrics(groundTruth, predicted)

        self.__dice()
        self.__F1(groundTruth, predicted)
        self.__FNR()
        self.__FPR()
        self.__IoU(groundTruth, predicted)
        self.__precision(groundTruth, predicted)
        self.__recall(groundTruth, predicted)
        self.__PWC()
        self.__specificity()

    def printIterationResults(self):

        return [self.diceList[-1], self.f1List[-1], self.fnrList[-1], self.fprList[-1], self.iouList[-1], self.precisionList[-1], self.recallList[-1], self.specificityList[-1]]

    def results(self, path):
        dice = np.mean(self.diceList)
        f1 = np.mean(self.f1List)
        fnr = np.mean(self.fnrList)
        fpr = np.mean(self.fprList)
        iou = np.mean(self.iouList)
        precision = np.mean(self.precisionList)
        recall = np.mean(self.recallList)
        spec = np.mean(self.specificityList)
        print("[dice f1 fnr fpr iou precision recall spec]")
        file = open(str(path) + "results.txt", 'w')
        file.write("\n****** Results *****\n")
        file.write("\nPath: " + str(path) + "\n")
        file.write("\nDice: {:.4f}".format(dice))
        file.write("\nF1: {:.4f}".format(f1))
        file.write("\nFNR: {:.4f}".format(fnr))
        file.write("\nFPR: {:.4f}".format(fpr))
        file.write("\nIoU: {:.4f}".format(iou))
        file.write("\nPrecision: {:.4f}".format(precision))
        file.write("\nRecall: {:.4f}".format(recall))
        file.write("\nSpec: {:.4f}".format(spec))
        return [dice, f1, fnr, fpr, iou, precision, recall, spec]
