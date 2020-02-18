%% Task 4.1
close all
lambda = 0.1;

what = lasso_ccd(t,X,lambda);
y = X*what;

plot(n,t,'o','MarkerSize',10);
hold on;
plot(n,y,'x','MarkerSize',10);
hold on
plot(ninterp,Xinterp*what)
xlabel('time');

set(gca, 'FontSize', 13);
legend('test value', 'reconstructed value','Interpolation');
title('lambda equal to 3')

%% Task 4.2 
lambda = 0.1;


counter_vec = [];
for i=1:20
    counter = 0;
    what = lasso_ccd(t,X,lambda);
    for k = 1:length(what)
        if what(k)~=0
            counter= counter + 1;
        end
    end
    counter_vec(i) = counter;
end
disp(counter)
mean(counter_vec)