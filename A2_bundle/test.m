clear all
load 'A2_data.mat'
X = train_data_01;
K = 3;
[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
C2 = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
%test = norm(C1-C2);
d = fxdist(X(:,1),C);
d2 = fcdist(C,C2);

y = step_assign_cluster(X,C);
hej1 = X(:,y==1);
mean_test = mean(hej1,2);




function d = fxdist(x,C) % Will return a vector of length N
    d = [];
    [M,N] = size(C);
    for i=1:N
       d(i) = norm(x-C(:,i)); 
    end
end

function d = fcdist(C1,C2) % Will return a scalar
    d = norm(C1-C2);
    %d = [];
end

function y = step_assign_cluster(X,C) % Each sample of X will get a label from 1:K (the nbr of clusters)
    [~,N] = size(X);
    %d = zeros(N,1);
    %d=[];
    y = zeros(N,1);
    
    for i=1:N
        d = fxdist(X(:,i),C);
        [~,I] = min(d);
        y(i) = I;
    end
end

