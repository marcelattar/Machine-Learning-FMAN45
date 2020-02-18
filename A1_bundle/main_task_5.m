%% Task 5.1
clear all
close all
load('A1_data.mat')

N_lambda = 100;
lambdamax = max(abs(X'*t));
lambdamin = 10^(-4);
K = 10;
lambdavec = exp( linspace( log(lambdamin), log(lambdamax), N_lambda));


[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdavec,K);

% Ploting
plot(log(lambdavec),RMSEval,'-o','MarkerSize',10);
hold on
plot(log(lambdavec),RMSEest,'-x','MarkerSize',10);
hold on
y1=get(gca,'ylim');
plot([log(lambdaopt) log(lambdaopt)],y1,'-','MarkerSize',10);
xlabel('log(\lambda)');

set(gca, 'FontSize', 13);
legend('RMSEval', 'RMSEest','Optimal \lambda = 2');
%% Task 5.2
clear all
close all
load('A1_data.mat')

N_lambda = 30;
lambdamax = max(abs(X'*t));
lambdamin = 1;
K = 10;
lambdavec = exp( linspace( log(lambdamin), log(lambdamax), N_lambda));

[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t,X,lambdavec,K);

what = lasso_ccd(t,X,lambdaopt);
y = X*what;

plot(n,t,'o','MarkerSize',10);
hold on;
plot(n,y,'x','MarkerSize',10);
hold on
plot(ninterp,Xinterp*what)
xlabel('time');

set(gca, 'FontSize', 13);
legend('test value', 'reconstructed value','Interpolation');
title('lambda equal to 1.9395')