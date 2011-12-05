addFoldersToPath;

[assignedLabels,totalCorrect,percentClassCorrect,confusionMat,pointData,partitionSet,learnedParameters] = ...
    doCrossValidationFixedParams('hw5-data/oakland_part3_am_rf.node_features', @oneVsAllSVM, {0.8}, 20);

avgCorrect = mean(totalCorrect)
avgCorrectPerClass = mean(percentClassCorrect,2)
