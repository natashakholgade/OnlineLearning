function Kinvtrain=GPRTrainOnlineTwoclass(Xtrain,lambda,kernelFunc,params)

ntrain=size(Xtrain,2);
%Kinvtrain=zeros(ntrain,ntrain);
%stepsize=50;
stepsize=ntrain;

%Kinvtrain(1:stepsize,1:stepsize)=eye(stepsize)/(kernelFunc(Xtrain(:,1:step
%size),Xtrain(:,1:stepsize),params)+lambda*lambda*eye(stepsize));
Kinvtrain=eye(stepsize)/(kernelFunc(Xtrain(:,1:stepsize),Xtrain(:,1:stepsize),params)+lambda*lambda*eye(stepsize));

% for i=stepsize+1:stepsize:ntrain
%     fprintf('%d...',i);
%     nsize=stepsize; idx=i:i+stepsize-1;
%     if ntrain<i+stepsize-1
%         idx=i:ntrain;
%         nsize=length(idx);        
%     end
%     
%     if mod(i-stepsize-1,40)==0, fprintf('\n'); end
%     Xonlinetrain=Xtrain(:,1:i-1);
%     %Fonlinetrain=Ftrain(1:i-1);
%     Xonline=Xtrain(:,idx);
%     
%     % compute matrix inverse incrementally
%     k=kernelFunc(Xonlinetrain,Xonline,params);
%     kappa=kernelFunc(Xonline,Xonline,params)+lambda*lambda*eye(nsize);
%     
%     Kinvprev=Kinvtrain(1:i-1,1:i-1);
%     kappabar=kappa-k'*Kinvprev*k;
%     
%     Kinvtrain(1:i-1,1:i-1)=Kinvprev+(Kinvprev*((k/kappabar)*k')*Kinvprev);
%     
%     kinv=-(Kinvprev*k)/kappabar;
%     Kinvtrain(1:i-1,idx)=kinv;
%     Kinvtrain(idx,1:i-1)=kinv';
%     Kinvtrain(idx,idx)=eye(nsize)/kappabar;
%     
%     norm(vec(Kinvtrain(1:i+stepsize-1,1:i+stepsize-1))...
%     -vec(inv(kernelExp(Xtrain(:,1:i+stepsize-1),Xtrain(:,1:i+stepsize-1),params)+lambda*lambda*eye(length(1:i+stepsize-1)))))
%     
% end

end