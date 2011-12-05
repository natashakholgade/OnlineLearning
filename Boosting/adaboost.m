function [W,Alpha]=adaboost(X,Y)
T=100;
N=size(X,2);
D=1/N*ones(1,N);
f=zeros(1,size(X,2));

W=zeros(size(X,1)+1,T);
Alpha=zeros(1,T);

for t=1:T
    [wt,e]=optimalDecisionStump(X,Y,D);
    
    %abs(0.5-e)
    ht=sign(wt'*[X;ones(1,N)]);
    alphat=.5*log((1-e)/e);
    f=f+alphat*ht;
    W(:,t)=wt;
    Alpha(t)=alphat;
    toplot=true;
    if (toplot)
        subplot(1,2,1);
        Xpos=X(:,sign(f)==Y);
        Xneg=X(:,sign(f)~=Y);
        plot(Xpos(1,:),Xpos(2,:),'g.'); hold on;
        plot(Xneg(1,:),Xneg(2,:),'r.');
        if (wt(1)>0)
            plot(-wt(end)*ones(1,2),[-1,1],'b-');
        else
            plot([-1,1],-wt(end)*ones(1,2),'b-');
        end
        drawnow;
    end
    D=D.*exp(-alphat*Y.*ht);
    D=D/sum(D);    
end

end

function [wt,e]=optimalDecisionStump(X,Y,D)

nsearch=200;
Er=zeros(size(X,1),nsearch);
E=eye(size(X,1));

for j=1:size(X,1)
    dsearch=(max(X(j,:))-min(X(j,:)))/(nsearch-1);
    theta=min(X(j,:)):dsearch:max(X(j,:));
    for k=1:length(theta)
        w_x=X(j,:)+theta(k);
        h_x=sign(w_x);
        Er(j,k)=sum(D.*(Y~=h_x));
    end    
end
subplot(1,2,2);
hold off; plot(abs(0.5-Er(1,:)),'b-'); hold on; 
plot(abs(0.5-Er(2,:)),'r-'); 
axis([0,size(Er,2),0,.1]);
[j,idx]=find(abs(0.5-Er)==max(abs(0.5-Er(:))));
j=j(round(end/2));
idx=idx(round(end/2));
dsearch=(max(X(j,:))-min(X(j,:)))/(nsearch-1);
theta=min(X(j,:)):dsearch:max(X(j,:));
wt=[E(:,j);theta(idx)];
e=Er(j,idx);

end