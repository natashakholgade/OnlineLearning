% Visualize 3D points colored per class
% visualizeClasses(pts, labels)
%   pts     3xN - 3D points
%   labels  1xN - point labels
function visualizeClasses(pts, labels)

classes = unique(labels);
nClasses = length(classes);

color = rand(nClasses, 3);

figure;
hold on;
for i=1:nClasses
    classId = classes(i);
    ptsInClass = pts(:,labels == classId);
    scatter3(ptsInClass(1,:), ptsInClass(2,:), ptsInClass(3,:), ...
             1, color(i,:));
end

axis equal; axis vis3d;
view([-13, 26]);

end