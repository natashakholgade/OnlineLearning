%[ptsactual,idactual,labelsactual,featuresactual]=readFile('hw5-data/oakland_part3_am_rf.node_features');
[ptsactual,idactual,labelsactual,featuresactual]=readFile('hw5-data/oakland_part3_an_rf.node_features');
labelsactual=double(labelsactual);
featuresactual=featuresactual(1:end-1,:);

Ts=[20:20:160];
errless=exp(log(.0001):(log(1)-log(.0001))/7:log(1));

nums=randperm(size(featuresactual,2));
features=featuresactual(:,nums);
labels=labelsactual(nums);
id=idactual(nums);
pts=ptsactual(:,nums);

trainnums=1:2:size(features,2);
testnums =2:2:size(features,2);


labelstrain=labels(trainnums);
labelstest=labels(testnums);


labelIDs=unique(labelstrain);
traincounts=zeros(size(labelIDs));
testcounts=zeros(size(labelIDs));
for i=1:length(labelIDs)
    traincounts(i)=sum(labelstrain==labelIDs(i));
    testcounts(i)=sum(labelstest==labelIDs(i));
end

[mtrain,imtrain]=min(traincounts);
[mtest,imtest]=min(testcounts);

trainnumsuse=zeros(mtrain*length(labelIDs),1);
testnumsuse=zeros(mtest*length(labelIDs),1);

for i=1:length(labelIDs)
    ftrain=find(labelstrain==labelIDs(i));
    ftrain=ftrain(1:mtrain);
    trainnumsuse((i-1)*mtrain+1:i*mtrain)=trainnums(ftrain);
    
    ftest=find(labelstest==labelIDs(i));
    ftest=ftest(1:mtest);
    testnumsuse((i-1)*mtest+1:i*mtest)=testnums(ftest);
end


Wsuse1=cell(length(Ts),length(errless));
Alphasuse1=cell(length(Ts),length(errless));
Percentcorrect1=zeros(length(Ts),length(errless));

Wsuse2=cell(length(Ts),length(errless));
Alphasuse2=cell(length(Ts),length(errless));
Percentcorrect2=zeros(length(Ts),length(errless));

for i=1:length(Ts)
    for j=1:length(errless)
        fprintf('T=%d, errless=%d\n',Ts(i),errless(j));
        params={Ts(i),errless(j)};
        fprintf('Fold1\n');
        [~,totalcorrecti,~,~,out]=BoostOneVersusRest(features,labels,trainnumsuse,testnums,params);
        Percentcorrect1(i,j)=totalcorrecti/length(testnums);
        Wsuse1{i,j}=out{1}; Alphasuse1{i,j}=out{2};
        fprintf('Fold2\n');
        [~,totalcorrecti,~,~,out]=BoostOneVersusRest(features,labels,testnumsuse,trainnums,params);
        Percentcorrect2(i,j)=totalcorrecti/length(trainnums);
        Wsuse2{i,j}=out{1}; Alphasuse2{i,j}=out{2};
    end
end


[r1,c1]=find(Percentcorrect1==max(Percentcorrect1(:)));
[r2,c2]=find(Percentcorrect2==max(Percentcorrect2(:)));

if max(Percentcorrect1(:))>max(Percentcorrect2(:))
    r=r1; c=c1;
else
    r=r2; c=c2;
end

T=Ts(r);
errles=errless(c);

save boost_crossval_an
