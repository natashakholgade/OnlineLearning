% Do cross validation with fixed parameters
% [classification,totalCorrect,percentClassCorrect,confusionMat] = ...
%   doCrossValidation(dataFile, fAlgorithm, params)
%   dataFile                            - path to data file
%   fAlgorithm                          - function handle to algorithm
%   algoParams                          - fixed algorithm params
%   numPartitions                       - number of data partitions
%   assignedLabels    {numPartitionsx1} - labels assiged to test data points (points depend on the partition)
%   totalCorrect        numPartitionsx1 - total percentage of correct classifications
%   percentClassCorrect KxnumPartitions - percentage of correct classificaitons per class and partition
%   confusionMat      KxKxnumPartitions - confusion matrix per partition
%   pointData                           - Point Cloud data loaded from dataFile
%   partitionSet      {numPartitions,1} - Partition Set of size numPartitions
%   learnedParameters {numPartitions,1} - Learned parameters per data partition
function [assignedLabels,totalCorrect,percentClassCorrect,confusionMat,pointData,partitionSet,learnedParameters] = ...
    doCrossValidationFixedParams(dataFile, fAlgorithm, algoParams, numPartitions)

pointData = PointCloudData(dataFile,1);
[partitionSet,n] = DataPartition.RandomDataPartitionSet(pointData, numPartitions, 0);

assignedLabels = cell(numPartitions,1);
totalCorrect=zeros(numPartitions,1);
percentClassCorrect=zeros(pointData.numClasses, numPartitions);
confusionMat=zeros(pointData.numClasses,pointData.numClasses,numPartitions);
learnedParameters = cell(numPartitions,1);

for i=1:n

    [assigned_i,correct_i,percentClass_i,confusion_i,learned_i]= ...
        fAlgorithm(pointData,partitionSet{i},algoParams);
    assignedLabels{i} = assigned_i;
    totalCorrect(i)=correct_i;
    percentClassCorrect(:,i)=percentClass_i;
    confusionMat(:,:,i)=confusion_i;
    learnedParameters{i} = learned_i;
    
end

end