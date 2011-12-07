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
      source        % file
      
   end
   
   methods
       
       % Constructor
       % obj = PointCloudData(dataFile)
       %    fataFile  - path to data file
       %    obj       - new point cloud data object  
       function obj = PointCloudData(dataFile, normalizeFeat)
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
              obj.source = {dataFile};
              
              % normalize features
              if (nargin > 1 && normalizeFeat)
                 % min/max except for bias term
                 maxFeat = max(obj.features(1:end-1,:), [], 2);
                 minFeat = min(obj.features(1:end-1,:), [], 2);
                 obj.features = [bsxfun(@rdivide, bsxfun(@minus, obj.features(1:end-1,:), minFeat), maxFeat - minFeat); ones(1,obj.numPts)];
              end
           else
              obj.pts = [];
              obj.numPts = 0;
              obj.labels = [];
              obj.classes = [];
              obj.numClasses = 0;
              obj.features = [];
              obj.numFeatures = 0;
              obj.ids = [];
           end
       end
       
       function obj = addRandomNoisyFeatures(obj, n) 
            newFeat = randn(n,obj.numPts);
            maxFeat = max(newFeat, [], 2);
            minFeat = min(newFeat, [], 2);
            newFeat = [bsxfun(@rdivide, bsxfun(@minus, newFeat, minFeat), maxFeat - minFeat); ones(1,obj.numPts)];

            obj.features = [obj.features(1:end-1,:); newFeat];
            obj.numFeatures = size(obj.features,1);
            
       end

       function obj = addCorruptedNoisyFeatures(obj, n) 
            corrupted = repmat(obj.features+randn(size(obj.features)),n,1);
            maxFeat = max(corrupted, [], 2);
            minFeat = min(corrupted, [], 2);
            corrupted = [bsxfun(@rdivide, bsxfun(@minus, corrupted, minFeat), maxFeat - minFeat); ones(1,obj.numPts)];
        
            obj.features = [obj.features(1:end-1,:); corrupted];
            obj.numFeatures = size(obj.features,1);
       end

       
   end
   
   methods(Static)
       
       function merged = MergePointClouds(p1, p2)
          if nargin >= 2
              
              if (length(intersect(p1.classes, p2.classes)) ~= p1.numClasses)
                 error('Point Cloud Data do not share the same classes.');   
              elseif (p1.numFeatures ~= p2.numFeatures)
                 error('Point Cloud Data do not share the same number of features.');   
              else

              merged = PointCloudData();
              merged.pts = [p1.pts p2.pts];
              merged.numPts = p1.numPts + p2.numPts;
              merged.labels = [p1.labels p2.labels];
              merged.classes = p1.classes;
              merged.numClasses = p1.numClasses;
              merged.features = [p1.features p2.features];
              merged.numFeatures = p1.numFeatures;
              merged.ids = [p1.ids p2.ids];
              merged.source = [p1.source; p2.source];
              
              end
              
          else
              error('Incorrect number of input arguments. Two Point Data Clouds should be provided.'); 
          end
       end
       
       
   end
end