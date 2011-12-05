%% Ideal result
idealWeights = [-1; 1; 0]; 
idealSeparator = @(x) (-idealWeights(1)*x + idealWeights(3))*(1/idealWeights(2));

%% Set up Point Cloud Data
xRange = 100;
pointData = PointCloudData();
pointData.numPts = 100;
pointData.pts = xRange*rand(2,pointData.numPts);
% fix points that lie on the linear separator
inLine = idealWeights'*[pointData.pts; ones(1,pointData.numPts)] == 0;
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
figure(); hold on;
scatter(positiveExamples(1,:), positiveExamples(2,:),5,'r')
scatter(negativeExamples(1,:), negativeExamples(2,:),5,'b')
plot(1:xRange, idealSeparator(1:xRange),'k--'); 

%% Create testing and training set
numPartitions = 2;
[partitionSet,n] = DataPartition.RandomDataPartitionSet(pointData, numPartitions);

%% Cross validation
assignedLabels = cell(numPartitions,1);
totalCorrect=zeros(numPartitions,1);
percentClassCorrect=zeros(pointData.numClasses, numPartitions);
confusionMat=zeros(pointData.numClasses,pointData.numClasses,numPartitions);
learnedParameters = cell(numPartitions,1);

for i=1:n

    [assigned_i,correct_i,percentClass_i,confusion_i,learned_i]= ...
        oneVsAllSVM(pointData,partitionSet{i},{0.85});
    assignedLabels{i} = assigned_i;
    totalCorrect(i)=correct_i;
    percentClassCorrect(:,i)=percentClass_i;
    confusionMat(:,:,i)=confusion_i;
    learnedParameters{i} = learned_i;
    
end

%% Print results
avgCorrect = mean(totalCorrect)
avgCorrectPerClass = mean(percentClassCorrect,2)

%% Plot results
for p=1:numPartitions
    color = rand(3,1);
    yPartition = @(x) (-learned_i(1,p)*x + learned_i(3,p))*(1/learned_i(2,p));
    plot(1:xRange, idealSeparator(1:xRange),':','color',color,'linewidth',2); 
end