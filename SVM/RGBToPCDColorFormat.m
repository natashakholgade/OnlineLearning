% s = RGBToPCDColorFormat(r, g, b)
%   color  3x1 - [red; green; blue] color components (in [0,255])
%   c   - color in PCD v.7 format
%
% The PCD color is computed following the description in:
% http://docs.pointclouds.org/trunk/structpcl_1_1_r_g_b.html
function c = RGBToPCDColorFormat(color)

% In C we would have 
% (int) c = ((int)r) << 16 | ((int)g) << 8 | ((int)b);

color = uint32(color);
r = typecast(color(1),'uint32');
g = typecast(color(2),'uint32');
b = typecast(color(3),'uint32');
c = bitor(bitshift(r,16),bitor(bitshift(g,8), b));

%% check by unpacking
%  int rgb = 4.101e+06;
%  uint8_t r = (rgb >> 16) & 0x0000ff;
%  uint8_t g = (rgb >> 8)  & 0x0000ff;
%  uint8_t b = (rgb)     & 0x0000ff;
% 
% c = 4.2108e+06;
% mask = hex2dec('000000FF');
% rgb = typecast(c,'uint32');
% 
% r = uint8(bitand(bitshift(rgb,-16),mask));
% g = uint8(bitand(bitshift(rgb,-8),mask));
% b = uint8(bitand(rgb,mask));
% rgb = [r g b]