load train_GPR_am
[ptstest,idtest,labelstest,featurestest]=readFile('hw5-data/oakland_part3_an_rf.node_features');
%load train_GPR_an
%[ptstest,idtest,labelstest,featurestest]=readFile('hw5-data/oakland_part3_am_rf.node_features');

labelstest=double(labelstest);
featurestest=featurestest(1:end-1,:);

nums=randperm(size(featuresactual,2));
features=featuresactual(:,nums);
labels=labelsactual(nums);
id=idactual(nums);
pts=ptsactual(:,nums);

trainnums=1:size(features,2);
labelstrain=labels(trainnums);

labelIDs=unique(labelstrain);
traincounts=zeros(size(labelIDs));
for i=1:length(labelIDs)
    traincounts(i)=sum(labelstrain==labelIDs(i));
end

[mtrain,imtrain]=min(traincounts);

trainnumsuse=zeros(mtrain*length(labelIDs),1);

for i=1:length(labelIDs)
    ftrain=find(labelstrain==labelIDs(i));
    ftrain=ftrain(1:mtrain);
    trainnumsuse((i-1)*mtrain+1:i*mtrain)=trainnums(ftrain);
    
end

testnums=size(features,2)+(1:size(featurestest,2));

params={lambdause,sigmause};
[Idxdecision,totalcorrect,percentperclasscorrect,confusionmatrix,vtrain]=GPROneVersusRest([features,featurestest],[labels,labelstest],trainnumsuse,testnums,params);

save final_train_am_test_an
%save final_train_an_test_am