function [TP FP FN TN SE stats] = confusionMatrixToVar(confusionMatrix, path)
    TP = confusionMatrix(1);
    FP = confusionMatrix(2);
    FN = confusionMatrix(3);
    TN = confusionMatrix(4);
    SE = confusionMatrix(5);
    
    recall = TP / (TP + FN);
    specficity = TN / (TN + FP);
    FPR = FP / (FP + TN);
    FNR = FN / (TP + FN);
    PBC = 100.0 * (FN + FP) / (TP + FP + FN + TN);
    precision = TP / (TP + FP);
    FMeasure = 2.0 * (recall * precision) / (recall + precision);
    
    stats = [recall specficity FPR FNR PBC precision FMeasure];
    f = fopen([path '/cm.txt'], 'wt');
    fprintf(f, '\nRecall : %1.5f',stats(1));
    fprintf(f, '\nSpecificity : %1.5f',stats(2));
    fprintf(f, '\nFPR : %1.5f',stats(3));
    fprintf(f, '\nFNR : %1.5f',stats(4));
    fprintf(f, '\nPBC : %1.5f',stats(5));
    fprintf(f, '\nPrecision : %1.5f',stats(6));
    fprintf(f, '\nFMeasure : %1.5f',stats(7));
    fclose(f);
end