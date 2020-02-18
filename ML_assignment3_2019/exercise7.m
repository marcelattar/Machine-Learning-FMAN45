%% Exerise 6
% Plot the filters the first convolutional layer learns. 
close all
clear all
load('models/cifar10_baseline.mat')
path1 = '/figures_fig/';
path2 = '/figures_png/';

first_conv_layer_weights = net.layers{2}.params.weights;
first_conv_layer_weights = rescale(first_conv_layer_weights);

FigH = figure('Position', get(0, 'Screensize'));

k = 1;
for i = 1:21
    subplot(3,7, k)
    imshow(first_conv_layer_weights(:, :, :, i))
    set(gca,'FontSize',20)
    title(['Filter ', num2str(k)])
    k = k + 1;
end

filename1 = [path1, 'E7_filters.fig'];
filename2 = [path2, 'E7_filters.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');

%% Plot a few images that are misclassified.

addpath(genpath('./'));
[x_train, y_train, x_test, y_test, classes] = load_cifar10(5);

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
for i=1:9
    subplot(3,3, k);
    idx = misclass_ind(i);
    imagesc(x_test(:,:,:,idx)/255); % Don't forget to divide by 255
    title([classes(y_test(idx)), classes(misclass(i))]);
    set(gca,'FontSize',20)
    axis off;
    k = k + 1;
end
filename1 = [path1, 'E7_misclass.fig'];
filename2 = [path2, 'E7_misclass.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');

%fprintf('Accuracy on the test set: %f\n', mean(vec(pred) == vec(y_test)));

%% Plot the confusion matrix for the predictions on the test set and 
 

conf_mat = confusionmat(y_test, uint8(pred));
FigH = figure('Position', get(0, 'Screensize'));
confusionchart(conf_mat, classes)
set(gca, 'FontSize', 20);
filename1 = [path1, 'E7_confusionchart.fig'];
filename2 = [path2, 'E7_confusionchart.png'];
saveas(FigH,[pwd filename1])
print(FigH,[pwd filename2],'-dpng','-r100');

% compute the precision and the recall for all digits.
PR = sum(conf_mat,2);
AR = sum(conf_mat);
TP = diag(conf_mat);

precision = TP./AR';
recall = TP./PR;

% Write down the number of parameters for all layers in the network. 
