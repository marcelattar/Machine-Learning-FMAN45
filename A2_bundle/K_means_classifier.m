function cluster_label = K_means_classifier(X,X_label,y,C,K)
    
    cluster_label = zeros(K,1);
    
    % I now count number of zeros and ones in each cluster, then I'm
    % looking at which is the biggest and assigning the largest value as
    % the cluster label.
    for i=1:K
        nbr_of_zeros = sum(X_label(y==i)==0); % count the nbr of zero-labeled samples in the cluster
        nbr_of_ones = sum(X_label(y==i)==1); % count the nbr of one-labeled samples in the cluster
        if nbr_of_zeros < nbr_of_ones
            cluster_label(i) = 1;
        else
            cluster_label(i) = 0;
        end
    end
    
end