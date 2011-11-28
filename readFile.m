function [pts,id,labels,features]=readFile(infile)

fid=fopen(infile);
fgets(fid);fgets(fid); fgets(fid);
C=textscan(fid,'%f %f %f %d %d %f %f %f %f %f %f %f %f %f %f');
pts=[C{1},C{2},C{3}]';
id=[C{4}]';
labels=[C{5}]';
features=[C{6:end}]';


end