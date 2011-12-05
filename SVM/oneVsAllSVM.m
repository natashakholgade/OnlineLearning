% Train and test One-Vs-All SVM for given partition
% [assigedLabels,totalCorrect,percentClassCorrect,confusionMat,W] = ...
%    oneVsAllSVM(pointData, partition, params)
%   pointData                     - Point Cloud data loaded from dataFile
%   partition                     - Data Partition with train and test identifiers
%   params                  {1x1} - svm parameters: lambda
%   assignedLabels           Nx1  - test point classifications
%   totalCorrect                  - percentage of correct classifications
%   percentClassCorrect      Kx1  - percentage of correct classificaitons per class
%   confusionMat             KxK  - confusion matrix
%   W                        FxK  - learned feature weights per class
function [assignedLabels,totalCorrect,percentClassCorrect,confusionMat,W] = ...
    oneVsAllSVM(pointData, partition, params)

% extract parameters and precomputed values
lambda = params{1};
trainPtIds = partition.trainPtIds;
testPtIds = partition.testPtIds;

%% Train: Update one-vs-all weights for each new data point
equalWeights = ones(pointData.numFeatures, 1).*(1/pointData.numFeatures);
W = repmat(equalWeights, 1, pointData.numClasses);

for pt = 1:partition.trainSize;
   ptId = trainPtIds(pt);
   alpha = 1/(lambda*(pt+1)); % typical 1/(lambda*t) alpha value
   y = (pointData.classes == pointData.labels(ptId)).*2 - 1; % set y in {-1,1}
   W = updateWeights(pointData.features(:,ptId), y, W, alpha);
end

%% Test: Classify and compute statistics
numCorrectClass = zeros(pointData.numClasses,1);
percentClassCorrect = zeros(pointData.numClasses,1);
confusionMat = zeros(pointData.numClasses,pointData.numClasses);

% classify test data
testY = pointData.labels(testPtIds)';
testFeat = pointData.features(:,testPtIds);
classScores = W'*testFeat;
[~,assignedClassPos] = max(classScores,[],1);
assignedLabels = pointData.classes(assignedClassPos');

% compute correct percentage per class and confusion matrix
correct = testY == assignedLabels;
for k = 1:pointData.numClasses
    
    ptsInClass = (testY == pointData.classes(k));
    numCorrectClass(k) = sum(correct == 1 & testY == pointData.classes(k));
    percentClassCorrect(k) = numCorrectClass(k)/sum(ptsInClass);
    
    for c = 1:pointData.numClasses
        confusionMat(k,c) = sum(ptsInClass & assignedLabels == pointData.classes(c));
    end
end
confusionMat=bsxfun(@rdivide,confusionMat,sum(confusionMat,2));

totalCorrect = sum(numCorrectClass)/partition.testSize;

end

% Update weights per class using subgradient
function updatedW = updateWeights(feat, y, W, alpha)
    
updatedW = zeros(size(W));
nClasses = size(W,2);
for k=1:nClasses
    
    subgrad = W(:,k);
    if (y(k)*(W(:,k)'*feat) < 1), 
        subgrad = subgrad - y(k)*feat; 
    end
    updatedW(:,k) = W(:,k) - alpha*subgrad;
    sizeW = norm(updatedW(:,k));
    updatedW(:,k) = updatedW(:,k) ./ sizeW;
end

end