function [muPosterior,Kposterior,Kjoint]=GPposterior(X,F,Xstar,muPrior,lambda,Kinit,kernelFunc,kernelFuncParams)

% X is a training matrix of size d x n (d = number of features, 
%     n=number of training examples)
% F is a 1 x n vector of target values
% xstar is the input test example (let's just allow this to be
%      d x ntest, but in the online learning case, it would
%      be d x 1)
% muPrior: d x ntest vector of means obtained from the prior mean
%     function at xstar (set to empty if you want GPposterior to estimate
%     muPrior)
% Kinit: covariance of X (with the noise). If you want GPposterior to compute, K, 
%      set Kinit to empty
% kernelFunc: function handle to kernel function, e.g. @kernelExp
%      lambda: for noise
% kernelFuncParams: cell array of for kernel function params
%
% Return:
% muPosterior - expected posterior value of Fstar
% KPosterior - posterior covariance matrix of Fstar

Q=lambda*lambda*eye(size(X,2));

nstar=size(Xstar,2);
if isempty(muPrior), muPrior=repmat(mean(F),1,nstar); end
if size(muPrior,2)==1, muPrior=repmat(muPrior,1,nstar); end
if isempty(Kinit), K=kernelFunc(X,X,kernelFuncParams)+Q; else K=Kinit; end
rank(K)
k=kernelFunc(X,Xstar,kernelFuncParams);
kappa=kernelFunc(Xstar,Xstar,kernelFuncParams)+lambda*lambda;
Kjoint=[K,k;k',kappa];

muPosterior=(muPrior'+(k'/K)*F')';
Kposterior=kappa+(k'/K)*k;

end