% Workspace should have:
%       - features and labels (as output by readFile)
%       - idTest (as output by splitDataIntoTrainAndTest)
%       - W and classes (as output by oneVsAllSVM or generated by trainSVM)

%% Parameters
generatePCD = 1;


%% Get data dimensions
featTest = features(:,idTest);
labTest = labels(:,idTest);
nTestPts = size(featTest,2);


%% Classify data
display(sprintf('Running classifySVM...'));
t = tic;
[assignedLabels, totalError, perClassError] = ...
    classifySVM(W, classes, featTest, labTest);
timeSpent = toc(t);
display(sprintf('Elapsed time: %.2f seconds (after classifying %d points)\n', ...
                timeSpent, size(featTest,2)));

            
%% Display results
display('Percentage of misclassifications per class:');
nClasses = size(classes,1);
for k=1:nClasses
    classId = classes(k);
    ptsInClass = find(labels(:,idTest) == classId);
    numPtsInClass = length(ptsInClass);
    display(sprintf(' [Class %d with %6d points] %.2f', ...
            classes(k), numPtsInClass, perClassError(k)));
end
display(sprintf('Percentage of total misclassifications: %.2f ~ %d/%d', ...
                 totalError, round(totalError*nTestPts), nTestPts));
             
             
%% Generate PCD file
% 4.2108e+06 blue
% 4.808e+06 green
% 
if (generatePCD == 1)

    classColors = rand(3,nClasses).*255;
    matched = classes(assignedLabels) == labTest';
    colors = zeros(3,nTestPts);
    for p=1:nTestPts
        pClass = find(classes == labTest(p));
        factor = (matched(p) + 1)*0.5;
        colors(:,p) = classColors(:,pClass).*factor;
    end
    generatePCDFile('svmResult.pcd', pts(:,idTest), colors);
    display('Saved PCD output.');
    
end