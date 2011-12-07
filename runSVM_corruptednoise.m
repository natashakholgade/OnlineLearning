close all; clear all;

dataFiles = {'hw5-data/oakland_part3_am_rf.node_features';
         'hw5-data/oakland_part3_an_rf.node_features'};
fileAbrv = {'am'; 'an'};
outputFolder = 'SVM/logs';
summaryLog = sprintf('%s/fitting.log', outputFolder);
finalLog = sprintf('%s/finalresults.log', outputFolder);

%% Parameters for SVM: lambda, t0
%  such that alpha = c/(lambda*(t+t0))
possibleLambda = [0.01; 0.001; 0.0001; 0.0001];   % lambda
possibleC = [1;5];
possibleT0 = [10; 100; 500]; % t0
      
%% Fixed params
numPartitions = 2; % number of partitions for training file to fit params
fAlgorithm = @SVM;

%% Derived params
trainResults = cell(2,1);
testResults = cell(2,1);

fidSummary = fopen(summaryLog,'w');
fidFinal = fopen(finalLog,'w');
    
for f=1:2

    % Pick files to test and test
    if (f == 1)
       trainFile = dataFiles{1};
       testFile = dataFiles{2};
    else
       trainFile = dataFiles{2};
       testFile = dataFiles{1};
    end
    
    %% Train
    bestParamsSVM = {};
    bestPercentCorrectSVM = 0;
    bestW = [];
    
    % Find best set of params
    for l=1:length(possibleLambda)
     for c=1:length(possibleC)
       for t=1:length(possibleT0)
           
           params = {possibleLambda(l), possibleC(c), possibleT0(t)};
           fprintf('{%f %f %f}, %s, ', params{1}, params{2}, params{3}, fileAbrv{f});
           
           mytic = tic();
           [assignedLabels,totalCorrect,percentClassCorrect,...
            confusionMat,pointData,partitionSet,learnedParameters] = ...
                doCrossValidationFixedParams_corruptednoise(trainFile, fAlgorithm, params, numPartitions, 2);
           processingTime = toc(mytic);
           
           [bestPartitionResult,bestPartition] = max(totalCorrect);
           fprintf('%f, %f\n', processingTime, bestPartitionResult);

           if bestPercentCorrectSVM < bestPartitionResult,
               bestParamsSVM = params;
               bestPercentCorrectSVM = bestPartitionResult;
               bestW = learnedParameters{bestPartition};
           end
            
           save(sprintf('%s/Train%d-Result-%s.mat',outputFolder, f,fileAbrv{f}), ...
                'assignedLabels','totalCorrect','percentClassCorrect', ...
                'confusionMat','pointData','partitionSet','learnedParameters', ...
                'bestPartitionResult','bestPartition','trainFile', 'testFile');

           logStr = sprintf('%s | %s | %f | %s | %f \n', ...
                            mat2str(cell2mat(params)), fileAbrv{f}, ...
                            processingTime, mat2str(totalCorrect), ...
                            bestPartitionResult);
           fprintf(fidSummary, '%s', logStr);
           pause(.01);
                        
       end
     end
    end
    
    save(sprintf('%s/Train%d-BestResult-%s.mat',outputFolder, f,fileAbrv{f}), ...
                'bestParamsSVM','bestPercentCorrectSVM','trainFile', 'testFile');
    
    % Train with everything and test in the other file
    fprintf('>> Final train with %s, %s, ', mat2str(cell2mat(bestParamsSVM)), fileAbrv{f});
    
%     pointDataTrain = PointCloudData(trainFile,1);
%     pointDataTest = PointCloudData(testFile,1);
%     allPointData = PointCloudData.MergePointClouds(pointDataTrain, pointDataTest);
%     allDataPartition = DataPartition((1:pointDataTrain.numPts)', ...
%                                      (pointDataTrain.numPts+1:pointDataTrain.numPts+pointDataTest.numPts)');
%     
%     mytic = tic();
%     [assignedLabels,totalCorrect,percentClassCorrect,confusionMat,finalW] = ...
%         SVM(allPointData, allDataPartition, bestParamsSVM);
%     processingTime = toc(mytic);
           
    pointDataTest = PointCloudData(testFile,1);
    pointDataTest = pointDataTest.addCorruptedNoisyFeatures(2);
    mytic = tic();
    
        testFeat = pointDataTest.features;
        testLabels = pointDataTest.labels';

        numCorrectClass = zeros(pointDataTest.numClasses,1);
        percentClassCorrect = zeros(pointDataTest.numClasses,1);
        confusionMat = zeros(pointDataTest.numClasses,pointDataTest.numClasses);

        % classify test data
        classScores = bestW'*testFeat;
        [~,assignedClassPos] = max(classScores,[],1);
        assignedLabels = pointData.classes(assignedClassPos');

        % compute correct percentage per class and confusion matrix
        correct = testLabels == assignedLabels;
        for k = 1:pointDataTest.numClasses,
            ptsInClass = (testLabels == pointDataTest.classes(k));
            numCorrectClass(k) = sum(correct == 1 & testLabels == pointDataTest.classes(k));
            percentClassCorrect(k) = numCorrectClass(k)/sum(ptsInClass);

            for k2 = 1:pointDataTest.numClasses
                confusionMat(k,k2) = sum(ptsInClass & assignedLabels == pointDataTest.classes(k2));
            end
        end
        confusionMat=bsxfun(@rdivide,confusionMat,sum(confusionMat,2));

        totalCorrect = sum(numCorrectClass)/pointDataTest.numPts;
    
    processingTime = toc(mytic);

    fprintf('%f, %f\n', processingTime, totalCorrect);
    
   save(sprintf('%s/Test%d-Result-%s.mat',outputFolder, f,fileAbrv{f}), ...
             'bestParamsSVM','assignedLabels','totalCorrect','percentClassCorrect', ...
             'confusionMat','bestW','pointDataTest', ...
             'trainFile', 'testFile');
    logStr = sprintf('%s | %s | %s | %f | %f | %s | %s | %s\n', ...
                     trainFile, testFile, mat2str(cell2mat(bestParamsSVM)), ...
                     processingTime, totalCorrect, mat2str(percentClassCorrect), ...
                     mat2str(confusionMat), mat2str(bestW));

%     save(sprintf('%s/Test%d-Result-%s.mat',outputFolder, f,fileAbrv{f}), ...
%                  'bestParamsSVM','assignedLabels','totalCorrect','percentClassCorrect', ...
%                  'confusionMat','finalW','pointDataTrain','pointDataTest','allPointData','allDataPartition', ...
%                  'trainFile', 'testFile');
%     logStr = sprintf('%s | %s | %s | %f | %s | %f | %s | %s\n', ...
%                      trainFile, testFile, mat2str(cell2mat(bestParamsSVM)), ...
%                      processingTime, mat2str(finalW), totalCorrect, mat2str(percentClassCorrect), ...
%                      confusionMat);
    fprintf(fidFinal, '%s', logStr);
             

end

fclose(fidSummary);
fclose(fidFinal);
