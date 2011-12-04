% Do cross validation with fixed parameters
% [classification,totalCorrect,percentClassCorrect,confusionMat] = ...
%   doCrossValidation(dataFile, fAlgorithm, params)
%   dataFile                            - path to data file
%   fAlgorithm                          - function handle to algorithm
%   algoParams                          - fixed algorithm params
%   numPartitions                       - number of data partitions
%   assignedLabels      NxnumPartitions - labeled assiged to data points
%   totalCorrect        numPartitionsx1 - total percentage of correct classifications
%   percentClassCorrect KxnumPartitions - percentage of correct classificaitons per class and partition
%   confusionMat      KxKxnumPartitions - confusion matrix per partition
%   pointData                           - Point Cloud data loaded from dataFile
%   partitionSet        {numPartitions} - Partition Set of size numPartitions
function [assignedLabels,totalCorrect,percentClassCorrect,confusionMat,pointData,partitionSet] = ...
    doCrossValidationFixedParams(dataFile, fAlgorithm, algoParams, numPartitions)

pointData = PointCloudData(dataFile);
[partitionSet,n] = DataPartition.RandomDataPartitionSet(pointData, numPartitions);

assignedLabels = zeros(pointData.numPts,numPartitions);
totalCorrect=zeros(numPartitions,1);
percentClassCorrect=zeros(pointData.numClasses, numPartitions);
confusionMat=zeros(pointData.numClasses,pointData.numClasses,numPartitions);

for i=1:n

    [assigned_i,correct_i,percentClass_i,confusion_i]= ...
        fAlgorithm(pointData,partitionSet{i},algoParams);
    assignedLabels(:,i) = assigned_i;
    totalCorrect(i)=correct_i;
    percentClassCorrect(:,i)=percentClass_i;
    confusionMat(:,:,i)=confusion_i;
end

end