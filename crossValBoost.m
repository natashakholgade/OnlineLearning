[pts,id,labels,features]=readFile('hw5-data/oakland_part3_am_rf.node_features');
labels=double(labels);
features=features(1:end-1,:);

Ts=[20:20:160];
errless=exp(log(.0001):(log(1)-log(.0001))/7:log(1));

trainnums=1:2:size(features,2);
testnums =2:2:size(features,2);

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
        [~,totalcorrecti,~,~,out]=BoostOneVersusRest(features,labels,trainnums,testnums,params);
        Percentcorrect1(i,j)=totalcorrecti/length(testnums);
        Wsuse1{i,j}=out{1}; Alphasuse1{i,j}=out{2};
        fprintf('Fold2\n');
        [~,totalcorrecti,~,~,out]=BoostOneVersusRest(features,labels,testnums,trainnums,params);
        Percentcorrect2(i,j)=totalcorrecti/length(trainnums);
        Wsuse2{i,j}=out{1}; Alphasuse2{i,j}=out{2};
    end
end