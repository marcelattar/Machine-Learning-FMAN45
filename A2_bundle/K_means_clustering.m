function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax

    % Step 1: Assign to clusters
    %y = [];
    y = step_assign_cluster(X,C);
    
    % Step 2: Assign new clusters
    %C = [];
    C = step_compute_mean(y,X,K);
    test = fcdist(C,Cold);
    if fcdist(C,Cold) < conv_tol
        %disp(test)
        %disp(kiter)
        return
    end
    Cold = C;
end

end


function d = fxdist(x,C) 

% Calculates the distance (2-norm) between the sample and all the centroids
% Input:        x  A column vector, 784x1, (one sample)
%               C  A matrix, 784xN (all centroids)
%
% Output:       d  Returns a vector, 1xN

    d = [];
    [~,N] = size(C);
    for i=1:N
       d(i) = norm(x-C(:,i)); 
    end
end

function d = fcdist(C1,C2)
% Calculates the distance (2-norm) between two sets of centroids
% Input:        C1 A matrix, 784xN (all centroids)
%               C2 A matrix, 784xN (all centroids)
%
% Output:       d  Returns a scalar.

    d = norm(C1-C2);
end

function y = step_assign_cluster(X,C) 
% Calculates a label from 1:K (the nbr of clusters) for each sample
% Input:        X  All samples, 784xN
%               C  A matrix, all centroids
%
% Output:       y  A vector with labels, Nx1

    [~,N] = size(X);
    y = zeros(N,1);
    
    for i=1:N
        d = fxdist(X(:,i),C); % Calculate the distance between each sample and each cluster
        [~,I] = min(d); 
        y(i) = I; % The 
    end
end

function C_new = step_compute_mean(y,X,K) 
% Returns K new centroids
% Input:        y  A vector with labels, Nx1
%               X  All samples, 784xN
%               K  A scalar, the number of centroids
% 
% Output:       C_new  A matrix, 784xK, with new centroids


    C_new = [];
    for i=1:K
        cluster_i = X(:,y==i); % Extract the coordinates in each cluster
        C_new = [C_new,mean(cluster_i,2)]; % Calculate a new centroid for each cluster
    end
end