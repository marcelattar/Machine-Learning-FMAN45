%% Task 6
clear all
close all
load('A1_data.mat')

% Deciding parameters
N_lambda = 100;
lambdamax = 1;
lambdamin = 10^(-4);
K = 10;
lambdavec = exp( linspace( log(lambdamin), log(lambdamax), N_lambda)); 

% Running main task
tic;
[Wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain,Xaudio,lambdavec,K);
toc;

%% 
% Ploting
%h = axes;
%set(h,'xscale','log')

plot(log(lambdavec),RMSEval,'-o','MarkerSize',10);
hold on
plot(log(lambdavec),RMSEest,'-x','MarkerSize',10);
hold on
y1=get(gca,'ylim');
plot([log(lambdaopt) log(lambdaopt)],y1,'-','MarkerSize',10);


xlabel('log(\lambda)');

set(gca, 'FontSize', 13);
legend('RMSEval', 'RMSEest','Optimal \lambda = 0.0045');

%% Task 7
%soundsc([Ttrain;Ttest],fs)
%soundsc(Ttest,fs)
[Yclean] = lasso_denoise(Ttest,Xaudio,lambdaopt);
save('denoised_audio','Yclean','fs');
%soundsc(Yclean,fs)
