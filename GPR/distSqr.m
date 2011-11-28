function [sqrdist] = distSqr(A,B)
% DISTSQR - Returns the squared distance for all pairs of points from set A
%           to set B
%
% Input:
%   A - DxM matrix - M points with D dimensions
%   B - DxN matrix - N points with D dimensions
%
% Output:
%   sqrdist - MxN matrix of squared distances
%
% Edward Hsiao
% edhsiao@cmu.edu
% 
% Modified by Natasha Kholgade to use bsxfun

%sqX1sqY1 = repmat(sum(A.^2)',[1,size(B,2)]);    % X1^2 + Y1^2
%sqX2sqY2 = repmat(sum(B.^2),[size(A,2),1]);     % X2^2 + Y2^2
X1X2Y1Y2 = -2*A'*B;                                % X1X2 + Y1Y2


% X1^2 + Y1^2) + (X2^2 + Y2^2) - 2*(X1X2 + Y1Y2)
%sqrdist = sqX1sqY1 + sqX2sqY2 - 2*X1X2Y1Y2;
sqrdist=bsxfun(@plus,X1X2Y1Y2,sum(A.^2,1)');
sqrdist=bsxfun(@plus,sqrdist,sum(B.^2,1));

