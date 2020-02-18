%% Exerise 6
% Plot the filters the first convolutional layer learns. 
close all
clear all
load('models/network_trained_with_momentum.mat')
path1 = '/figures_fig/';
path2 = '/figures_png/';

first_conv_layer_weights = net.layers{2}.params.weights;

FigH = figure('Position', get(0, 'Screensize'));

k = 1;
for i = 1:16
    subplot(4,4, k)
    imshow(first_conv_layer_weights(:, :, :, i))
    set(gca,'FontSize',20)
    title(['Filter ', num2str(k)])
    k = k + 1;
end

filename1 = [path1, 'E6_filters.fig'];
filename2 = [path2, 'E6_filters.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');

%% Plot a few images that are misclassified.
% Evaluate on the test set. There are 10000 images, so it takes some time
addpath(genpath('./'));
load('models/network_trained_with_momentum.mat')
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');
y_test(y_test==0) = 10;
x_test = reshape(x_test, [28, 28, 1, 10000]);

pred = zeros(numel(y_test),1);
batch = 16;
for i=1:batch:size(y_test)
    idx = i:min(i+batch-1, numel(y_test));
    % note that y_test is only used for the loss and not the prediction
    y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

misclass_ind = find(pred ~= y_test);
misclass = pred(misclass_ind); % These are the misclassifications

%%
FigH = figure('Position', get(0, 'Screensize'));
k = 1;
classes = [1:9 0];
for i=1:16
    subplot(4,4, k);
    idx = misclass_ind(i);
    imagesc(x_test(:,:,:,idx));
    colormap(gray);
    title(['Ground truth:', num2str(classes(y_test(idx))),' Prediction: ', num2str(classes(misclass(i)))]);
    set(gca,'FontSize',20)
    axis off;
    k = k + 1;
end
filename1 = [path1, 'E6_misclass.fig'];
filename2 = [path2, 'E6_misclass.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');

%fprintf('Accuracy on the test set: %f\n', mean(vec(pred) == vec(y_test)));

%% Plot the confusion matrix for the predictions on the test set and 
 

conf_mat = confusionmat(y_test, pred);
FigH = figure('Position', get(0, 'Screensize'));
confusionchart(conf_mat)
set(gca, 'FontSize', 20);
filename1 = [path1, 'confusionchart.fig'];
filename2 = [path2, 'confusionchart.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');


% compute the precision and the recall for all digits.
PR = sum(conf_mat,2);
AR = sum(conf_mat);
TP = diag(conf_mat);

precision = TP./AR';
recall = TP./PR;

% Write down the number of parameters for all layers in the network. 
