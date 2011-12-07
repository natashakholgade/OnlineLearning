function saveConfusionMatrixImage(C,filename)

C=bsxfun(@rdivide,C,sum(C,2));
axes('position',[0,0,1,1]);
imagesc(C); colormap(gray);
axis equal; axis off;
saveas(gca,filename);

end