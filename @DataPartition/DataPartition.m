% Data partition object
% Properties:
%   trainSize  - size of training set
%   trainPtIds - position of data points that belong to the train set
%   testSize   - size of testing set
%   testPtIds  - position of data points that belong to the test set
classdef DataPartition
   properties
      trainSize     % number of data points in train set
      trainPtIds    % position of data points that belong to train set
      testSize      % number of data points in test set
      testPtIds     % position of data points that belong to test set
   end
   methods
       
       % Constructor
       function obj = DataPartition(train, test)
           if  nargin == 0
               obj.trainPtIds = [];
               obj.trainSize = 0;
               obj.testPtIds = [];
               obj.testSize = 0;
           elseif  nargin == 2
               obj.trainPtIds = train;
               obj.trainSize = length(train);
               obj.testPtIds = test;
               obj.testSize = length(test);
           else
               error('Incorrect number of input arguments. Train and test data should be provided, or none.'); 
           end
       end
       
   end
   
   methods(Static)
       
       % Generate random data partition set (for leave-one-out
       % corssvalidation)
       % [s,count] = RandomDataPartitionSet(pointData, nPartitions, equalProportionTraining)
       %    pointData   - PointCloudData (with point ids)
       %    nPartitions - number of partitions of the data 
       %    equalProportionTraining - {0,1} training set should have uniform proportion of samples from the different classes (optional)
       %    s           - cell (set) of DataPartitions
       %    count       - number of partitions (set size)
       %
       % Generates a random permutation of the data, and assigns one
       % partition to training and the rest to testing. The different
       % possible choises of training data leads to a set of data 
       % partitions.
       function [s, count] = RandomDataPartitionSet(pointData, nPartitions, equalProportionTraining)
           
           if  nargin ~= 2 &&  nargin ~= 3
               error('Incorrect number of input arguments. Point cloud data and number of partitions should be provided.'); 
           end
           
           if nargin < 3, equalProportionTraining = 0; end
           
           nPts = size(pointData.pts,2);
           dataPositions = 1:nPts;
           randomizedPositions = randperm(nPts)';
           partitionSize = ceil(nPts/nPartitions);
           
           s = cell(nPartitions,1);
           count = 0;
           for i=1:partitionSize:nPts
               
              % split data por partition i 
              idxtrain = i:i+partitionSize-1;
              if (i+partitionSize-1 > nPts)
                  idxtrain = i:nPts;
              end
              idxtest  = setdiff(dataPositions,idxtrain);
              train = randomizedPositions(idxtrain);
              test = randomizedPositions(idxtest);
              
              % sanity check
              assert(length(train) + length(test) == nPts);
              
              if equalProportionTraining,
                 
                  ptsInClass = cell(pointData.numClasses,1);
                  numPtsInClass = zeros(pointData.numClasses,1);
                  classLabels = pointData.labels(train);
                  for k=1:pointData.numClasses
                     ptsInClass{k} =  train(classLabels == pointData.classes(k));
                     numPtsInClass(k) = length(ptsInClass{k});
                  end
                  numPerClass = min(numPtsInClass);
                  reducedTrain = [];
                  for k=1:pointData.numClasses
                      reducedTrain = [reducedTrain ptsInClass{k}(1:numPerClass)'];
                  end
                  train = reducedTrain';
              end
              
              count = count + 1;
              s{count} = DataPartition(train, test);
           end
       end
       
       % f = DisplayPartition(pointData, partition, colorTrain, colorTest)
       %    pointData  - Point Cloud Data
       %    partition  - Data Partition
       %    colorTrain - train data color
       %    colorTest  - test data color
       function f = DisplayPartition(pointData, partition, colorTrain, colorTest)
           
           if  nargin == 2
                colorTrain = [1 0 0];
                colorTest = [0 0 1];
           elseif nargin ~= 4
               error('Incorrect number of input arguments. Point cloud data, partition, train color and test color should be provided.');                
           end
           
           ptsTrain = pointData.pts(:,partition.trainPtIds);
           ptsTest = pointData.pts(:,partition.testPtIds);
           
           f = figure();
           hold on; grid on;
           scatter3(ptsTrain(1,:), ptsTrain(2,:), ptsTrain(3,:),2,colorTrain);
           scatter3(ptsTest(1,:), ptsTest(2,:), ptsTest(3,:),2,colorTest);
           axis equal; axis vis3d;
           
       end
   end
end