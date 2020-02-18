%% E1, ALWAYS RUN THIS ONE BEFORE RUNNING E2-E4
close all
clear all
load 'A2_data.mat'

% Preping the data
X = train_data_01;
%X = test_data_01;
[D,N] = size(X);
P = min(D,N);

% Normalizing X so that the mean is 0 for each column
X_norm = X - mean(X,2);

% Calculating U, the left singular vectors
%[U,eig_val] = eig(X_norm*X_norm');
[U,~,~] = svd(X_norm);

% Calculating the reduced matrix
d = 2;
U_new = [];
for i=1:d
    U_new = [U_new, U(:,i)];
end
X_new = X_norm'*U_new; % projecting on the principal components

% Ploting
figure(1);
plot(X_new(train_labels_01==0,1), X_new(train_labels_01==0,2),'r.',...,
    X_new(train_labels_01==1,1),X_new(train_labels_01==1,2),'b.','LineWidth', 3);
xlabel('First principal component');
ylabel('Second principal component');
lgd = legend('Class 1','Class 2');
lgd.FontSize = 15;
title 'PCA';
set(gca, 'FontSize', 13);


%% E2 plot, 2 clusters
K = 2;

[idx,centroid] = K_means_clustering(X,K);

for i=1:K
   centroid(:,i) = centroid(:,i)-mean(X,2); 
end

projected_centroid = U_new'*centroid; % Projecting the centroids onto the principal components

% Ploting the figure
figure(1);
plot(X_new(idx==1,1), X_new(idx==1,2),'r.',...,
    X_new(idx==2,1),X_new(idx==2,2),'b.',projected_centroid(1,:),projected_centroid(2,:),...,
    'kx','LineWidth',3);
xlabel('First principal component');
ylabel('Second principal component');
lgd = legend('Cluster 1','Cluster 2','Centroids');
lgd.FontSize = 15;
title 'Cluster Assignments';
set(gca, 'FontSize', 13);

%% E2 plot, 5 clusters
K = 5;

[idx,centroid] = K_means_clustering(X,K);

for i=1:K
   centroid(:,i) = centroid(:,i)-mean(X,2); 
end

projected_centroid = U_new'*centroid; % Projecting the centroids onto the principal components



% Ploting the figure
figure(1);
plot(X_new(idx==1,1), X_new(idx==1,2),'r.',...,
    X_new(idx==2,1),X_new(idx==2,2),'b.',...,
    X_new(idx==3,1), X_new(idx==3,2),'g.',...,
    X_new(idx==4,1), X_new(idx==4,2),'y.',...,
    X_new(idx==5,1), X_new(idx==5,2),'m.',...,
    projected_centroid(1,:),projected_centroid(2,:),'kx','LineWidth',3);
xlabel('First principal component');
ylabel('Second principal component');
lgd = legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Centroids');
lgd.FontSize = 15;
title 'Cluster Assignments';
set(gca, 'FontSize', 13);

%% E3, K=2
K=2;

[~,C] = K_means_clustering(X,K);

% Ploting

figure(1)

subplot(1,2,1)
imshow(reshape(C(:,1),28,28))
title 'Cluster 1';
set(gca, 'FontSize', 13);
subplot(1,2,2)
imshow(reshape(C(:,2),28,28))
title 'Cluster 2';
set(gca, 'FontSize', 13);

%% E3, K=5
K=5;

[~,C] = K_means_clustering(X,K);

% Ploting
figure(1)

subplot(1,5,1)
imshow(reshape(C(:,1),28,28))
title 'Cluster 1';
set(gca, 'FontSize', 13);

subplot(1,5,2)
imshow(reshape(C(:,2),28,28))
title 'Cluster 2';
set(gca, 'FontSize', 13);

subplot(1,5,3)
imshow(reshape(C(:,3),28,28))
title 'Cluster 3';
set(gca, 'FontSize', 13);

subplot(1,5,4)
imshow(reshape(C(:,4),28,28))
title 'Cluster 4';
set(gca, 'FontSize', 13);

subplot(1,5,5)
imshow(reshape(C(:,5),28,28))
title 'Cluster 5';
set(gca, 'FontSize', 13);

%% E4 a)
X = train_data_01;
X_label = train_labels_01;
K=5;
[y,C] = K_means_clustering(X,K);

cluster_label = K_means_classifier(X,X_label,y,C,K)

%% E4/E5 b)
% Train data
%X = train_data_01;
%X_label = train_labels_01;

% Test data
X = test_data_01;
X_label = test_labels_01;

K=5;
[y,C] = K_means_clustering(X,K);

cluster_label = K_means_classifier(X,X_label,y,C,K)

% The data needed for the table
label_one = zeros(K,1);
label_zero = zeros(K,1);
misclassified = zeros(K,1);
[~,N] = size(X); % N_train / N_test

for i=1:K
    [size_of_cluster,~] = size(find(y==i));
    
    nbr_of_zeros = sum(X_label(y==i)==0); % count the nbr of zero-labeled samples in the cluster
    nbr_of_ones = sum(X_label(y==i)==1); % count the nbr of one-labeled samples in the cluster
    
    label_zero(i) = nbr_of_zeros;
    label_one(i) = nbr_of_ones;
    
    if cluster_label(i)==1
        misclassified(i) = nbr_of_zeros;
    else
        misclassified(i) = nbr_of_ones;
    end
end

Sum_misclassified = sum(misclassified);
Misclassification_rate = Sum_misclassified/N*100; % The misclassification rate in percentage

%% E6
close all
clear all
load 'A2_data.mat'

% Train data
X = train_data_01';
T = train_labels_01;

% Test data
%X = test_data_01';
%T = test_labels_01;

model = fitcsvm(X,T);
predicted_labels = predict(model,X);

% This is the matrix from the A2 description
classification_matrix = zeros(2,2);

[N,~] = size(X); % N_train / N_test

% Top row
nbr_true_zero_pred = sum(T(predicted_labels==0)==0);
nbr_false_zero_pred = sum(T(predicted_labels==0)==1);

%Bottom row
nbr_false_one_pred = sum(T(predicted_labels==1)==0);
nbr_true_one_pred = sum(T(predicted_labels==1)==1);

% Filling in my matrix
classification_matrix(1,1) = nbr_true_zero_pred;
classification_matrix(1,2) = nbr_false_zero_pred;
classification_matrix(2,1) = nbr_false_one_pred;
classification_matrix(2,2) = nbr_true_one_pred;

sum_misclassified = classification_matrix(1,2) + classification_matrix(2,1);
Misclassification_rate = sum_misclassified/N;

%% E7

% Train data
%X = train_data_01';
%T = train_labels_01;

% Test data
X = test_data_01';
T = test_labels_01;

beta = 1;

model = fitcsvm(X,T,'KernelFunction','gaussian','KernelScale', beta);

predicted_labels = predict(model,X);

% This is the matrix from the A2 description
classification_matrix = zeros(2,2);

[N,~] = size(X); % N_train / N_test, remember that we transposed X before.

% Top row
nbr_true_zero_pred = sum(T(predicted_labels==0)==0);
nbr_false_zero_pred = sum(T(predicted_labels==0)==1);

%Bottom row
nbr_false_one_pred = sum(T(predicted_labels==1)==0);
nbr_true_one_pred = sum(T(predicted_labels==1)==1);

% Filling in my matrix
classification_matrix(1,1) = nbr_true_zero_pred;
classification_matrix(1,2) = nbr_false_zero_pred;
classification_matrix(2,1) = nbr_false_one_pred;
classification_matrix(2,2) = nbr_true_one_pred;

sum_misclassified = classification_matrix(1,2) + classification_matrix(2,1);
Misclassification_rate = sum_misclassified/N;