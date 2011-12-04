function [muPosterior]=GPRTestOnlineTwoClass(Xtrain,Ftrain,Xtest,KinvGPR,lambda,kernelFunc,params)

ntest=size(Xtest,2);
%Kinvtrain(1:stepsize,1:stepsize)=eye(stepsize)/(kernelFunc(Xtrain(:,1:stepsize),Xtrain(:,1:stepsize),params)+lambda*lambda*eye(stepsize));

D=KinvGPR*Ftrain';
muPrior=zeros(1,ntest);
muPosterior=zeros(1,ntest);
%KPosterior=zeros(ntest,ntest);

stepsize=1000;

for i=1:stepsize:ntest
idx=i:i+stepsize-1;
if i+stepsize-1>ntest
    idx=i:ntest;
end
nsize=(length(idx));
Xonline=Xtest(:,idx);
k=kernelFunc(Xtrain,Xonline,params);
%kappa=kernelFunc(Xonline,Xonline,params)+lambda*lambda*eye(nsize);
muPosterior(idx)=(muPrior(idx)'+k'*D)';
%if mod(i-1,20000)==0
%    fprintf('%d,',i);
%end
%KPosterior(idx,idx)=kappa-k'*KinvGPR*k;
end
fprintf('\n');

end