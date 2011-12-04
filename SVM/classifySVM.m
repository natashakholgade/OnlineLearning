% Classify test data and compute error proportions (per class and in total)
% [totalError, perClassError] = testSVM(W, classes, features, labels)
%   W               MxK - features weights learned for all K classes
%   classes         Kx1 - classes identifiers (ordered as in W)
%   features        MxN - test data features
%   labels          1xN - test data labels (must match classes)
%   assignedLabels  1xN - assiged label using W
%   totalError          - percentage of misclassifications
%   perClassError   Kx1 - percentage of misclassifications per class
function [assignedLabels, totalError, perClassError] = ...
    classifySVM(W, classes, features, labels)

nClasses = size(W,2);
nTestPoints = size(features, 2);
assignedLabels = zeros(1,nTestPoints);
misclassified = zeros(nClasses,1);
perClassError = zeros(nClasses,1);

% Classify keeping track of errors
for pt = 1:nTestPoints
    scores = W'*features(:,pt);
    assignedLabels(1,pt) = find(scores == max(scores));
    labelClass = find(classes == labels(pt));
    misclassified(labelClass) = misclassified(labelClass) + ...
                                    (assignedLabels(1,pt) ~= labelClass);
end

% Find proportion of errors
for k=1:nClasses
    classId = classes(k);
    ptsInClass = find(labels == classId);
    numPtsInClass = length(ptsInClass);
    perClassError(k) = misclassified(k)/numPtsInClass;
end

totalError = sum(misclassified)/nTestPoints;
