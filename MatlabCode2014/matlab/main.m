videoPath = '/home/ibrahim/Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/intermittentPan';
binaryFolder = '/home/ibrahim/Desktop/Dataset/Change Detection Dataset/dataset2014/dataset/PTZ/intermittentPan/mask-mcdFlow';


cm = processVideoFolder(videoPath,binaryFolder);

[TP FP FN TN SE stats] = confusionMatrixToVar(cm,binaryFolder)