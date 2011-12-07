addFoldersToPath;
close all; clear all;

[assignedLabels,totalCorrect,percentClassCorrect,confusionMat,pointData,partitionSet,learnedParameters] = ...
doCrossValidationFixedParams('hw5-data/oakland_part3_am_rf.node_features', @SVM, {0.00001, 1, 1000}, 2);

avgCorrect = mean(totalCorrect)
avgCorrectPerClass = mean(percentClassCorrect,2)