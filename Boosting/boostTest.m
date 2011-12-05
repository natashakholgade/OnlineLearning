function [f,y]=boostTest(X,W,Alpha)

f=Alpha*sign(W'*[X;ones(1,size(X,2))]);
y=sign(f);

end