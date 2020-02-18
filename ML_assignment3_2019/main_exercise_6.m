clear all
close all
load('models/network_trained_with_momentum.mat');

% load the dataset
x_train = loadMNISTImages('data/mnist/train-images.idx3-ubyte');
y_train = loadMNISTLabels('data/mnist/train-labels.idx1-ubyte');
perm = randperm(numel(y_train));
x_train = x_train(:,perm);
y_train = y_train(perm);
    
% note that matlab indexing starts at 1 but we should classify
% the digits 0-9, so 1-9 are correct while the digit 0 gets label 10.
y_train(y_train==0) = 10;
classes = [1:9 0];
    

% Selecting the first sample of x_train
X = x_train(:,1);

conv_layer_1 = net.layers(2);

