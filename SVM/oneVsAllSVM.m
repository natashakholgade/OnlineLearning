% [W, classes] = oneVsAllSVM(features, labels)
%   features    MxN - M features for N data points
%   labels      1xN - data point labels (K classes)
%   W           MxK - feature weights learned for all K classes
%   classes     Kx1 - classes identifiers (ordered as in W)
function [W, classes] = oneVsAllSVM(features, labels, lambda)

% Find number of classes
classes = unique(labels)';
nClasses = length(classes);

% Allocate necessary space
nFeatures = size(features,1);
W = repmat(ones(nFeatures, 1).*(1/nFeatures), 1, nClasses);

% Update one-vs-all weights for each new data point
nPts = size(features, 2);
for pt = 1:nPts
   alpha = 1/(lambda*(pt+1));
   y = (classes == labels(pt)).*2 - 1; % set y in {-1,1}
   W = updateWeights(features(:,pt), y, W, alpha);
end

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