% Group data into training and testing set
% [idTrain, idTest] = splitDataIntoTrainAndTest(labels, probTrain)
%   labels      1xN - points labels
%   probTrain       - probabilty of picking a point for the train set
%   idTrain     Kx1 - position of points selected for training
%   idTest      Mx1 - position of points selected for testing
%
% We split the data per class, and then group them all to form idTrain and
% idTest. The resulting arrays should satisfy K+M = N (all points should
% have been assigned a group).
%
% NOTE: Instead of using real data ids (as provided in the data files), we
% rely on the position of the points in the arrays output by readFile
function [idTrain, idTest] = ...
    splitDataIntoTrainAndTest(labels, probTrain)

% we want to split the set of points, so we need 0 < probTrain < 1
assert(probTrain > 0 && probTrain < 1);

idTrain = [];
idTest = [];

% process every class independently
classes = unique(labels);
nClasses = length(classes);

for i=1:nClasses
    
    % find points belonging to this class
    classId = classes(i);
    ptsInClass = find(labels == classId);
    numPtsInClass = length(ptsInClass);
    
    % flip a biased coin to group points
    coin = rand(numPtsInClass,1);
    train = [ptsInClass(coin <= probTrain)'];
    test = [ptsInClass(coin > probTrain)'];
    
    % show statistics
    display(sprintf('[Class %5d] %5d for train, %5d for test', ...
                    classId, length(train), length(test)));
    
    % save result
    idTrain = [idTrain; train];
    idTest = [idTest; test];
end
    
% make sure we got all the points
assert(length(idTrain) + length(idTest) == length(labels));

end