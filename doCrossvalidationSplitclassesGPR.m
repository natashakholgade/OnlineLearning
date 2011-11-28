[pts,id,labels,features]=readFile('hw5-data/oakland_part3_am_rf.node_features');
labels=double(labels);
load nums_randomlyrearranged;
features=features(1:end-1,:);
cd GPR

labels_rr=labels(nums);
features_rr=features(nums);

labelIDs=unique(labels);

for i=1:length(labelIDs)
    idx=labels==labelIDs;
    featuresi=features(:,idx);
    labelsi=labels(idx);
    
end