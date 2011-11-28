function K=kernelExp(Xi,Xj,params)
sigma=params{1}; % at most sigma is a vector representing a diagonal matrix
D=distSqr(bsxfun(@rdivide,Xi,sigma),bsxfun(@rdivide,Xj,sigma));
K=exp(-D/2);
end