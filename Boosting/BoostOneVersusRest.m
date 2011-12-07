function [imulticlassdecision,totalcorrect,percentclasscorrect,confusionmatrix,out]=BoostOneVersusRest(features,labels,trainnums,testnums,params)

featureoffset=min(features(:,trainnums),[],2);
featurescale=max(features(:,trainnums),[],2)-min(features(:,trainnums),[],2);

trainfeatures=bsxfun(@rdivide,bsxfun(@minus,features(:,trainnums),featureoffset),featurescale);
testfeatures=bsxfun(@rdivide,bsxfun(@minus,features(:,testnums),featureoffset),featurescale);

labelIDs=unique(labels);
ntrain=length(trainnums);
ntest=length(testnums);

twoclassdecisions=zeros(ntest,length(labelIDs));

Ws=cell(1,length(labelIDs));
Alphas=cell(1,length(labelIDs));

tic;
for i=1:length(labelIDs)
    l=labels(trainnums);
    y=2*double(l==labelIDs(i))-1;
    if exist('params','var')
        [W,Alpha]=boost(trainfeatures,y,params);
    else
        [W,Alpha]=boost(trainfeatures,y);
    end
    Ws{i}=W;
    Alphas{i}=Alpha;
end
toc;

tic;
for i=1:length(labelIDs)
    f=boostTest(testfeatures,Ws{i},Alphas{i});
    twoclassdecisions(:,i)=f;
end
toc;

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

out={Ws,Alphas};

end