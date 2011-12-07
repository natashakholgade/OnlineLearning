function prettyplot(ptstest,Idxdecision,prettycolors,filename)

colorsuse=prettycolors(Idxdecision,:);
scatter3(ptstest(1,:),ptstest(2,:),ptstest(3,:),5,colorsuse,'filled');axis vis3d; axis equal; view(-81.5,16);
saveas(gcf,filename);

end