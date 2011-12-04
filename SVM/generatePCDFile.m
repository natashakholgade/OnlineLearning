% Generate Point Cloud Data (v.7) file from 3D points and RGB colors
% generatePCDFile(ptsName, pts, colors)
%   pcdName         - output file name
%   pts         3xN - 3D points
%   colors      3xN - R-G-B colors per point
%
% NOTE: If a file pcdName already exists, it will be 
% overwritten!
function generatePCDFile(pcdName, pts, colors)

nPts = size(pts, 2);

fid = fopen(pcdName, 'w', 'n', 'ASCII');
fprintf(fid, '# .PCD v.7 - Point Cloud Data file format\n');
fprintf(fid, 'VERSION .7\n');
fprintf(fid, 'FIELDS x y z rgb\n');
fprintf(fid, 'SIZE 4 4 4 4\n');
fprintf(fid, 'TYPE F F F F\n');
fprintf(fid, 'COUNT 1 1 1 1\n');
fprintf(fid, 'WIDTH %d\n', nPts);
fprintf(fid, 'HEIGHT 1\n');
fprintf(fid, 'VIEWPOINT 0 0 0 1 0 0 0\n');
fprintf(fid, 'POINTS %d\n', nPts);
fprintf(fid, 'DATA ascii\n');
for p = 1:nPts
    floatCol = RGBToPCDColorFormat(colors(:,p));
    fprintf(fid, '%f %f %f %f\n', pts(1,p), pts(2,p), pts(3,p), floatCol);    
end
fclose(fid);
