function [W,Alpha]=boost(X,Y,params)

T=100;
errless=.001;

if exist('params','var')
    T=params{1};
    errless=params{2};
end

N=size(X,2);
toplot=false;
f=zeros(1,N);
W=zeros(size(X,1)+1,T);
Alpha=zeros(1,T);

for t=1:T
    [wt,p]=optimalDecisionStump(X,Y,f);
    fprintf('%f,',p);
    if mod(t,20)==0
        fprintf('\n');
    end
    ht=sign(wt'*[X;ones(1,N)]);
    idx=ht~=Y;
    epsratio=sum(exp(-Y(~idx).*f(~idx)),2)/sum(exp(-Y(idx).*f(idx)),2);
    %[p,1/(1+epsratio)]
    alphat=.5*log(epsratio);    
    f=f+alphat*ht;
    W(:,t)=wt;
    Alpha(t)=alphat;
        
    if (toplot)
        subplot(1,2,1);
        Xpos=X(:,sign(f)==Y);
        Xneg=X(:,sign(f)~=Y);
        plot(Xpos(1,:),Xpos(2,:),'g.'); hold on;
        plot(Xneg(1,:),Xneg(2,:),'r.');        
        %if (wt(1)>0)
        %    plot(-wt(end)*ones(1,2),[-1,1],'b-');
        %else
        %    plot([-1,1],-wt(end)*ones(1,2),'b-');
        %end
        drawnow;
    end
    if p<errless
        break;
    end
end

W=W(:,1:t);
Alpha=Alpha(1:t);

end

function [wt,p]=optimalDecisionStump(X,Y,f)
G=grad(Y,f);
N=size(X,2);
E=eye(size(X,1));

nsearch=200;
proj=zeros(size(X,1),nsearch);

for j=1:size(X,1)
    dsearch=(max(X(j,:))-min(X(j,:)))/(nsearch-1);
    theta=(-max(X(j,:)):dsearch:-min(X(j,:)));
    for k=1:length(theta)
        w_x=X(j,:)+theta(k);
        h_x=sign(w_x);
        proj_j=dotproduct(G,h_x)./sqrt(dotproduct(h_x,h_x));
        proj(j,k)=proj_j;
    end
end

proj=abs(proj);
%subplot(1,2,2);
%hold off; plot(proj(1,:),'b-'); hold on; 
%plot(proj(2,:),'r-'); 
%axis([0,size(proj,2),0,.1]);
[j,idx]=find(proj==max(proj(:)));
j=j(round(end/2));
idx=idx(round(end/2));
dsearch=(max(X(j,:))-min(X(j,:)))/(nsearch-1);
theta=(-max(X(j,:)):dsearch:-min(X(j,:)));
wt=[E(:,j);theta(idx)];
p=proj(j,idx);

end



function g=grad(Y,f)

g=-Y.*exp(-Y.*f);

end

function d=dotproduct(f_x,g_x)

N=size(f_x,2);
d=sum(bsxfun(@times,f_x,g_x),2)/N;

end