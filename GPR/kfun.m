function f=kfun(v,features,kernelfunc,params,lambda)

% f=Kv, each element of K is built from kernelfunc acting on features
f=zeros(size(v));
stepsize=500;
for i=1:stepsize:length(v)
    %fprintf('%d\n',i);
    idx=i:i+stepsize-1;
    if i+stepsize-1>length(v)
        idx=i:length(v);
    end
    f(idx)=(kernelfunc(features(:,idx),features,params)+lambda*lambda*eye(length(idx),length(v)))*v;
end

end