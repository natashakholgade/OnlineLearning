% Train and test SVM for given partition
% [assigedLabels,totalCorrect,percentClassCorrect,confusionMat,W] = ...
%    oneVsAllSVM(pointData, partition, params)
%   pointData                     - Point Cloud data loaded from dataFile
%   partition                     - Data Partition with train and test identifiers
%   params                  {3x1} - svm parameters: lambda, c, t0 such that alpha = c/(lambda*(t+to)
%   assignedLabels           Nx1  - test point classifications
%   totalCorrect                  - percentage of correct classifications
%   percentClassCorrect      Kx1  - percentage of correct classificaitons per class
%   confusionMat             KxK  - confusion matrix
%   W                        FxK  - learned feature weights per class
function [assignedLabels,totalCorrect,percentClassCorrect,confusionMat,W] = ...
    SVM(pointData, partition, params)

% extract parameters and precomputed values
lambda = params{1};
c = params{2};
t0 = params{3};
trainPtIds = partition.trainPtIds;
trainFeat = pointData.features(:,trainPtIds);
trainLabels = pointData.labels(trainPtIds);

%% Train: Update one-vs-all weights for each new data point
% equalWeights = ones(pointData.numFeatures, 1);
% W = repmat(equalWeights./norm(equalWeights), 1, pointData.numClasses);
W = zeros(pointData.numFeatures, pointData.numClasses);

for pt = 1:partition.trainSize,
   alpha = c/(lambda*(pt+t0)); 
   if (alpha < 1e-12)
      warning('alphs = %f for t = %d', alpha, pt);
   end
   y = (pointData.classes == trainLabels(pt)).*2 - 1; % set y in {-1,1}
   W = updateWeights(trainFeat(:,pt), y, W, alpha, lambda);
end
% alpha

if (partition.testSize > 0)
    
    %% Test: Classify and compute statistics
    testPtIds = partition.testPtIds;
    testFeat = pointData.features(:,testPtIds);
    testLabels = pointData.labels(testPtIds)';

    numCorrectClass = zeros(pointData.numClasses,1);
    percentClassCorrect = zeros(pointData.numClasses,1);
    confusionMat = zeros(pointData.numClasses,pointData.numClasses);

    % classify test data
    classScores = W'*testFeat;
    [~,assignedClassPos] = max(classScores,[],1);
    assignedLabels = pointData.classes(assignedClassPos');

    % compute correct percentage per class and confusion matrix
    correct = testLabels == assignedLabels;
    for k = 1:pointData.numClasses,
        ptsInClass = (testLabels == pointData.classes(k));
        numCorrectClass(k) = sum(correct == 1 & testLabels == pointData.classes(k));
        percentClassCorrect(k) = numCorrectClass(k)/sum(ptsInClass);

        for c = 1:pointData.numClasses
            confusionMat(k,c) = sum(ptsInClass & assignedLabels == pointData.classes(c));
        end
    end
    confusionMat=bsxfun(@rdivide,confusionMat,sum(confusionMat,2));

    totalCorrect = sum(numCorrectClass)/partition.testSize;
    
else
    
    %% no testing
    assignedLabels = [];
    totalCorrect = 0;
    percentClassCorrect = zeros(pointData.numClasses,1);
    confusionMat = zeros(pointData.numClasses,pointData.numClasses);
    
end

end

%% Update weights per class using subgradient
function updatedW = updateWeights(feat, y, W, alpha, lambda)
    
%% feat = feat./norm(feat);

updatedW = zeros(size(W));
nClasses = size(W,2);
for k=1:nClasses
    
    subgrad = W(:,k).*lambda;
    if (y(k)*(W(:,k)'*feat) < 1), 
        subgrad = subgrad - y(k)*feat; 
    end
    updatedW(:,k) = W(:,k) - subgrad.*alpha;
    size2W = updatedW(:,k)'*updatedW(:,k);
    if (size2W > 1/lambda)
        updatedW(:,k) = updatedW(:,k)./sqrt(size2W*lambda);
    end
end

end