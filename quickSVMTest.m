addFoldersToPath;
close all; clear all;

[assignedLabels,totalCorrect,percentClassCorrect,confusionMat,pointData,partitionSet,learnedParameters] = ...
    doCrossValidationFixedParams('hw5-data/oakland_part3_am_rf.node_features', @oneVsAllSVM, {0.2, 2, 10}, 60);

avgCorrect = mean(totalCorrect)
avgCorrectPerClass = mean(percentClassCorrect,2)
