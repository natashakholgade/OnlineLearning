[pts,id,labels,features]=readFile('hw5-data/oakland_part3_an_rf.node_features');
labels=double(labels);
load nums_randomlyrearranged2;
%features=features(1:end-1,:);
nrandfeatures=20;
%features=[features;randn(nrandfeatures,size(features,2))];
features=[features;features+randn(size(features));features+randn(size(features));features+randn(size(features))];
cd GPR

trainsize=ceil(length(nums)/20);

totalcorrect=zeros(20,1);
percentclasscorrect=zeros(20,5);
confusionmatrix=zeros(5,5,20);
numtestnums=zeros(20,1);

for i=1:trainsize:length(nums)
%i=trainsize+1;
    fprintf('%d,',i);
    idxtrain=i:i+trainsize-1;
    if i+trainsize-1>length(nums)
        idxtrain=i:length(nums);
    end
    idxtest=setdiff(1:length(nums),idxtrain);
    trainnums=nums(idxtrain);
    testnums=nums(idxtest);
    [Idxdecisioni,totalcorrecti,percentclasscorrecti,confusionmatrixi]=GPROneVersusRest(features,labels,trainnums,testnums);
    if i==1
       Idxdecision=Idxdecisioni;
    end
    %totalcorrecti
    %percentclasscorrecti
    %confusionmatrixi
    totalcorrect((i-1)/trainsize+1)=totalcorrecti;
    percentclasscorrect((i-1)/trainsize+1,:)=percentclasscorrecti;
    confusionmatrix(:,:,(i-1)/trainsize+1)=confusionmatrixi;
    numtestnums( (i-1)/trainsize+1)=length(testnums);
end

totalcorrect=totalcorrect./numtestnums;

cd ../

confusionmatrixnorm=bsxfun(@rdivide,confusionmatrix,sum(confusionmatrix,2));
meanconfusionmatrixnorm=mean(confusionmatrixnorm,3)

save an_results_noiseadded
clear