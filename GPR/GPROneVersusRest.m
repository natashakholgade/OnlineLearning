function [imulticlassdecision,totalcorrect,percentclasscorrect,confusionmatrix]=GPROneVersusRest(features,labels,trainnums,testnums,params)

kernelFunc=@kernelExp;
%sigma=(max(features(:,trainnums),[],2)-min(features(:,trainnums),[],2));
%sigma(sigma<1e-6)=.1;
%sigma=sigma/1;

% scale the features so they lie between 0 and 1
featureoffset=min(features(:,trainnums),[],2);
featurescale=max(features(:,trainnums),[],2)-min(features(:,trainnums),[],2);

trainfeatures=bsxfun(@rdivide,bsxfun(@minus,features(:,trainnums),featureoffset),featurescale);
testfeatures=bsxfun(@rdivide,bsxfun(@minus,features(:,testnums),featureoffset),featurescale);

if ~exist('params','var')
    lambda=.01;
    sigma=.5;
    kernelparams={sigma};
else
    lambda=params{1};
    kernelparams=params(2);
end

labelIDs=unique(labels);
ntrain=length(trainnums);
ntest=length(testnums);
%KinvGPR=zeros(ntrain,ntrain,length(labelIDs));

twoclassdecisions=zeros(ntest,length(labelIDs));
vtrain=zeros(ntrain,length(labelIDs));
for i=1:length(labelIDs)
    fprintf('Training %d\n',i);
    l=labels(trainnums);
    f=(2*double(l==labelIDs(i))-1)';
    vtrain(:,i)=GPRTrainOnlineTwoclass(trainfeatures,f,lambda,kernelFunc,kernelparams);
%    KinvGPR(:,:,i)=KinvGPRi;
end
for i=1:length(labelIDs)
    fprintf('Testing %d\n',i);
    d=GPRTestOnlineTwoClass(trainfeatures,testfeatures,vtrain(:,i),kernelFunc,kernelparams);
    twoclassdecisions(:,i)=d;
end
[~,imulticlassdecision]=max(twoclassdecisions,[],2);
outputtestlabels=labelIDs(imulticlassdecision);
realtestlabels=labels(testnums);

%[outputtestlabels;realtestlabels]
totalcorrect=sum(outputtestlabels==realtestlabels);

sumIDs=zeros(1,length(labelIDs));
numIDs=zeros(1,length(labelIDs));

confusionmatrix=zeros(length(labelIDs),length(labelIDs));
for i=1:length(labelIDs)
    idx=realtestlabels==labelIDs(i);
    numIDs(i)=sum(idx);
    sumIDs(i)=sum(outputtestlabels(idx)==realtestlabels(idx));
    for j=1:length(labelIDs)
        %idx1=outputtestlabels==labelIDs(j);
        %confusionmatrix(i,j)=sum(outputtestlabels(idx1)==realtestlabels(idx));
        confusionmatrix(i,j)=sum(realtestlabels==labelIDs(i) & outputtestlabels==labelIDs(j));
    end
end

percentclasscorrect=sumIDs./numIDs

end