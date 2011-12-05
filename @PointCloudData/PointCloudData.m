% Point Cloud Data 
% Properties:
%     pts         3xN - 3D points
%     numPts          - total number of points: N
%     labels      1xN - points labels
%     classes     Kx1 - classes
%     numClasses      - number of classes: K
%     features    FxN -  point features
%     numFeatures     - number of feature dimentions: F
%     ids           N - point unique identifiers
classdef PointCloudData
   properties
       
      pts           % N 3D points
      numPts        % total number of points: N
      labels        % 1xN points labels
      classes       % Kx1 classes
      numClasses    % number of classes: K
      features      % FxN point features
      numFeatures   % number of feature dimentions: F
      ids           % N point unique identifiers
      
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
              obj.classes = unique(obj.labels)';
              obj.numClasses = length(obj.classes);
              obj.features = ptFeatures;
              obj.numFeatures = size(obj.features,1);
              obj.ids = ptIdentifiers;
           else
               error('Not enough input arguments. Path to data file should be provided.'); 
           end
       end
       
   end
end