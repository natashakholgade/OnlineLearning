% Workspace should have:
%       - features and labels (as output by readFile)
%       - idTrain (as output by splitDataIntoTrainAndTest)

%% Parameters
lambda = 1;

%% Train
display(sprintf('Running oneVsAllSVM...'));
t = tic;
[W, classes] = oneVsAllSVM(features(:,idTrain), labels(:,idTrain),...
                           lambda);
timeSpent = toc(t);
display(sprintf('\ttraining took %.2f seconds.', timeSpent));

%% Display results
W