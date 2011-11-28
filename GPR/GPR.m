function GPR(features,labels,trainnums,testnums,sigma,lambda)

trainfeatures=features(:,trainnums);
testfeatures=features(:,testnums);
trainlabels=labels(trainnums);
testlabels=labels(testnums);
if ~exist('sigma','var')
    sigma=(max(trainfeatures,[],2)-min(trainfeatures,[],2))/8;
end
if ~exist('lambda','var')
    lambda=.001;
end
sigma(sigma<1e-3)=.1;
[muTest,KTest]=GPposterior(trainfeatures,trainlabels,testfeatures,mean(trainlabels),lambda,[],@kernelExp,{sigma});

muTest
testlabels

abs(muTest-testlabels)

end