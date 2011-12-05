close all;

%% Ideal result
idealWeights = [-1; 1; 0]; 
idealSeparator = @(x) (-idealWeights(1)*x - idealWeights(3))*(1/idealWeights(2));

%% Set up Point Cloud Data
xRange = 1;
pointData = PointCloudData();
pointData.numPts = 5000;
pointData.pts = xRange*rand(2,pointData.numPts);
% fix points that lie on the linear separator
inLine = abs(idealWeights'*[pointData.pts; ones(1,pointData.numPts)]) < 1e-6;
if (sum(inLine) > 0)
    pointData.pts(inLine) = pointData.pts(:,inLine) + repmat([0;2],1,length(inLine));
    display(sprintf('Fixed %d points that were on the ideal separator'));
end
pointData.labels = (idealWeights'*[pointData.pts; ones(1,pointData.numPts)] > 0)*2 - 1;
pointData.numClasses = 2;
pointData.classes = [-1;1];
pointData.features = [pointData.pts; ones(1,pointData.numPts)];
pointData.numFeatures = 3;
pointData.ids = 1:pointData.numPts;

%% Plot data
positiveExamples = pointData.pts(:,pointData.labels == 1);
negativeExamples = pointData.pts(:,pointData.labels == -1);

% figure(1); hold on;
% scatter(positiveExamples(1,:), positiveExamples(2,:),5,'r')
% scatter(negativeExamples(1,:), negativeExamples(2,:),5,'b')
% plot(0:0.1:xRange, idealSeparator(0:0.1:xRange),'k--'); 
% title('All data and ideal linear separator');
% axis equal;
% %pause

%% Create testing and training set
numPartitions = 3;
[partitionSet,n] = DataPartition.RandomDataPartitionSet(pointData, numPartitions);

%% Cross validation
assignedLabels = cell(numPartitions,1);
totalCorrect=zeros(numPartitions,1);
percentClassCorrect=zeros(pointData.numClasses, numPartitions);
confusionMat=zeros(pointData.numClasses,pointData.numClasses,numPartitions);
learnedParameters = cell(numPartitions,1);

for i=1:n

    [assigned_i,correct_i,percentClass_i,confusion_i,learned_i]= ...
        oneVsAllSVM(pointData,partitionSet{i},{0.3, 2, 10});
    assignedLabels{i} = assigned_i;
    totalCorrect(i)=correct_i;
    percentClassCorrect(:,i)=percentClass_i;
    confusionMat(:,:,i)=confusion_i;
    learnedParameters{i} = learned_i;
    
end

%% Print results
avgCorrect = mean(totalCorrect)
avgCorrectPerClass = mean(percentClassCorrect,2)
learnedParameters{:}

%% Plot results
figure(2);
for p=1:numPartitions
    color = [0; 0; 0]; %color = rand(3,1);

    w = learnedParameters{p};    
    trainPts = pointData.pts(:,partitionSet{p}.trainPtIds);
    trainLabels = pointData.labels(:,partitionSet{p}.trainPtIds);
    trainPos = trainPts(:,trainLabels == 1);
    trainNeg = trainPts(:,trainLabels == -1);
    testPts = pointData.pts(:,partitionSet{p}.testPtIds);
    testLabels = pointData.labels(:,partitionSet{p}.testPtIds);
    testPos = testPts(:,testLabels == 1);
    testNeg = testPts(:,testLabels == -1);
    
    subplot(numPartitions, size(w,2) + 1, 1 + (size(w,2) + 1)*(p-1)); hold on;
    scatter(trainPos(1,:), trainPos(2,:),2,'r');
    scatter(trainNeg(1,:), trainNeg(2,:),2,'b');
    title(sprintf('Partition %d, Training data',p));
    xlabel('x'); ylabel('y'); axis equal; axis([0 xRange 0 idealSeparator(xRange)]);
    
    for c=1:size(w,2)
        yPartition = @(x) (-w(1,c)*x - w(3,c))*(1/w(2,c));
        
        subplot(numPartitions, size(w,2) + 1, 1 + c + (size(w,2) + 1)*(p-1)); hold on;
        scatter(testPos(1,:), testPos(2,:),5,'r');
        scatter(testNeg(1,:), testNeg(2,:),5,'b');
        plot(0:0.1:xRange, yPartition(0:0.1:xRange),'-','color',color,'linewidth',2);
        title(sprintf('Class. %d with Testing Data', c));
        xlabel('x'); ylabel('y'); axis equal; %axis([0 xRange 0 idealSeparator(xRange)]);
        
    end
    
end