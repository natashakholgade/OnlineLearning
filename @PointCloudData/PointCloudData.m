classdef PointCloudData
   properties
      pts
      numPts
      labels
      features
      ids
      classes
      numClasses
   end
   methods
       
       % Constructor
       % obj = PointCloudData(dataFile)
       %    fataFile  - path to data file
       %    obj       - new point cloud data object  
       function obj = PointCloudData(dataFile)
           if  nargin > 0
              [pt,ptIdentifiers,ptLabels,ptFeatures]= ...
                  readFile(dataFile);
              obj.pts = pt;
              obj.numPts = size(obj.pts,2);
              obj.labels = double(ptLabels);
              obj.features = ptFeatures;
              obj.ids = ptIdentifiers;
              obj.classes = unique(obj.labels)';
              obj.numClasses = length(obj.classes);
           else
               error('Not enough input arguments. Path to data file should be provided.'); 
           end
       end
       
   end
end