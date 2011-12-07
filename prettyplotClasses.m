function prettyplotClasses(pointData,assignedLabels,prettycolors,filename)

Idxdecision = zeros(size(assignedLabels));
for k=1:pointData.numClasses
    inK = find(assignedLabels == pointData.classes(k));
    Idxdecision(inK) = k;
end

ptstest = pointData.pts;
colorsuse=prettycolors(Idxdecision,:);
scatter3(ptstest(1,:),ptstest(2,:),ptstest(3,:),5,colorsuse,'filled');axis vis3d; axis equal; view(-81.5,16);
view([-82 14]);
saveas(gcf,filename);

end