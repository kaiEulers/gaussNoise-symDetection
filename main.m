%% PRM Matlab Workshop 3: Gaussian Noisy Comm Channel & Symbol Detection
%% Question 4
% Generate Exponential Random Variable
clc, clear
ts = 1e-1;
min = 0;
max = 8;

% Parameter
lambda = 1;

% Empirical
% rand() ouputs a random number with distribution
% uniform(0,1)
trials = 30e3;
for k = 1:trials
    
    rSq_emp(k,1) = -1/lambda * log(1 - rand());
    theta_emp(k,1) = 2*pi*rand();
    
end

% Theoritical
rSq = min:ts:max-ts;
Fr = 1 - exp(-lambda.*rSq);

figure(1)
cdfplot(rSq_emp)
hold on
plot(rSq, Fr)
hold off
axis([min max 0 1.1])
grid on
xlabel('r'), ylabel('F_R(r)')
title('R^2 ~ Exponential(1)')
legend({'Empirical CDF', 'Theoretical CDF'})

n = 1:length(rSq_emp);
figure(10)
subplot(211)
scatter(n, rSq_emp, 1, 'fill')
ylabel('r^2'), title('R^2 ~ exponential(\lambda)')
subplot(212)
scatter(n, theta_emp, 1, 'fill')
ylabel('\theta'), title('\Theta ~ Uniform(0, 2\pi)')
%% Question 5a
% Gassian Noise Plot
clc, clear

% Parameter
lambda = 1;

trials = 30e3;
for k = 1:trials
    % Generate noise
    [n1 n2 r] = gaussianNoise(lambda);
    
    % Array of data from 30e3 trials
    R(k,1) = r;
    N1(k,1) = n1;
    N2(k,1) = n2;
end

% Joint probability plot of the first 1000 elements of
% n1_emp and n2_emp
figure(2)
scatter(N1(1:1000), N2(1:1000), 5, 'fill')
axis([-3 3 -3 3])
xlabel('n1'), ylabel('n2')
title('f_{N1,N2}(n1, n2)')
grid on

%% Question 5bi,ii,ii, 6, 7

% Question 5bi
% Empirical probability of Y1<0 given that X1=-1
clc, clear
lambda = 1;

trials = 30e3;
for k = 1:trials
    
    x1 = -1;
    [n1 n2] = gaussianNoise(lambda);
    y1(k,1) = x1 + n1;
    
end

% Sum of input x1 and noise n1 must be less than 0 for
% output y1 to be decoded as -1
X1crct = (y1 < 0);
% Probability of Y1<0 given that X1=-1
empProb_X1crt = mean(X1crct);


% Question 5bii
% Empirical probability of Y2>=0 given that X2=1
lambda = 1;

trials = 30e3;
for k = 1:trials
    
    x2 = 1;
    [n1 n2] = gaussianNoise(lambda);
    y2(k,1) = x2 + n2;
    
end

% Sum of input x2 and noise n2 must be more than or
% equal to 0 for output y1 to be decoded as -1
X2crct = (y2 >= 0);
% Probability of Y2>=0 given that X2=1
empProb_X2crt = mean(X2crct);


% Question 5biii
% Empirical probability of Y1<0 given that X1=-1 and
% Y2>=0 given that X2=1
lambda = 1;

trials = 30e3;
for k = 1:trials
    
    x1 = -1;
    x2 = 1;
    [n1 n2 r] = gaussianNoise(lambda);
    y1(k,1) = x1 + n1;
    y2(k,1) = x2 + n2;
    
end

% For both y1 and y2 to be decoded correctly, elements
% of A1(k) and A2(k) should both be 1.
X1nX2crct = (y1 < 0)&(y2 >= 0);
% Probability of Y1<0 given that X1=-1 and Y2>=0 given
% that X2=1
empProb_X1nX2crt = mean(X1nX2crct);
empProb_err = 1 - empProb_X1nX2crt;

% Question 6 & 7
% Theoritical Probability

% normcdf(x, mu, sigma)
% sigma of normcdf() is STANDARD DEVIATION, NOT VARIANCE!!!
% N~(mu, sigma^2) = N~(0, 0.5)
prob_X1crt = normcdf(1, 0, sqrt(0.5));
prob_X2crt = 1 - normcdf(-1, 0, sqrt(0.5));
prob_X1nX2crt = normcdf(1, 0, sqrt(0.5)) - normcdf(-1, 0, sqrt(0.5));
prob_err = 1 - prob_X1nX2crt;

% Comparing Empirical & Theoritical Probabilities
% If N1 and N2 are independent, P[A1]*P[A2] = P[A1 and A2]
Probability = {
    'P[X1 decoded correctly]'
    'P[X2 decoded correctly]'
    'P[X1 & X2 decoded correctly]'
    'P[X1 decoded correctly]*P[X2 decoded correctly]'
    'P[Error]'
    };
Empirical = [
    empProb_X1crt
    empProb_X2crt
    empProb_X1nX2crt
    empProb_X1crt*empProb_X2crt
    empProb_err
    ];
Theoretical = [
    prob_X1crt
    prob_X2crt
    prob_X1nX2crt
    prob_X1crt*prob_X2crt
    prob_err
    ];
disp(table(Probability, Empirical, Theoretical))
disp(' ')
disp('For both empirical and theoritical probabilities,')
disp('P[X1 decoded correctly]*P[X2 decoded correctly] = P[X1 & X2 decoded correctly]')
disp('Therefore gaussian noise N1 and N2 are independent of each other.')
%% Question 8a, 8b, 8c
% Scatter Plot of Y = X + N WITHOUT decision rule of
% all received Y and all errors of Ys
% Empirical probability of error
% X = {1+j, -1+j, -1-j, 1-j}
% N = Gaussian(0, 0.5) + j*Gaussian(0, 0.5)
clc, clear

lambda = 1;
trials = 10e3;
% Vector count variables for all four outcomes of y
k1 = 1; k2 = 1; k3 = 1; k4 = 1;
for k0 = 1:trials
    % Generate Noise
    [n1 n2] = gaussianNoise(lambda);
    n = n1 + j*n2;
    
    % Generate random vector x = x1 + j*x2
    die = rand();
    cnt = 1;
    if die <= 0.25
        % x = 1 + j, 1st quandrant
        x = 1 + j;
        % Store y = (1+n1) + j*(1+n2) in y{1} and
        % increment count variable k1
        y{1}(k1,1) = x + n;
        k1 = k1 + 1;
    elseif (die > 0.25) && (die <= 0.5)
        % x = -1 + j, 2nd quandrant
        x = -1 + j;
        % Store y = (-1+n1) + j*(1+n2) in y{2} and
        % increment count variable k2
        y{2}(k2,1) = x + n;
        k2 = k2 + 1;
    elseif (die > 0.5) && (die <= 0.75)
        % x = -1 - j, 3rd quandrant
        x = -1 - j;
        % Store y = (-1+n1) + j*(-1+n2) in y{3} and
        % increment count variable k3
        
        y{3}(k3,1) = x + n;
        k3 = k3 + 1;
    else
        % x = 1 - j, 4th quandrant
        x = 1 - j;
        % Store y = (1+n1) + j*(-1+n2) in y{4} and
        % increment count variable k4
        y{4}(k4,1) = x + n;
        k4 = k4 + 1;
    end
    cnt = cnt + 1;
end

% ---------------Scatter plot of all received signals
% Plot Axis
y1Axis = [-5 5];
y2Axis = [-5 5];
figure(3)
subplot(121)
scatter(real(y{1}), imag(y{1}), 5, 'fill')
hold on
scatter(real(y{2}), imag(y{2}), 5, 'fill')
scatter(real(y{3}), imag(y{3}), 5, 'fill')
scatter(real(y{4}), imag(y{4}), 5, 'fill')
hold off
axis([y1Axis y2Axis])
xlabel('y_1'), ylabel('y_2')
title({
    'Received Signals'
    'Y = X + N'
    ['Gaussian Noise \lambda = ' num2str(lambda) ', \sigma^2 = ' num2str(1/(2*lambda))]
    })
grid on
legend([
    'y =  1 + j + n'
    'y = -1 + j + n'
    'y = -1 - j + n'
    'y =  1 - j + n'
    ])

% --------------------Remove correctly decoded Ys
% Decision Rule Threshold
yThr = [0 0];
% yErr{1} holds all errors of y for x = 1 + j
yErr{1} = y{1};
k = 1;
while k <= length(yErr{1})
    if ( real(yErr{1}(k)) >= yThr(1) ) && ( imag(yErr{1}(k)) >= yThr(2) )
        % If y is decoded correctly, remove from the
        % vector
        yErr{1}(k) = [];
    else
        % Otherwise, move to the next element of the
        % vector
        k = k + 1;
    end
end

% yErr{2} holds all errors of y for x = -1 + j
yErr{2} = y{2};
k = 1;
while k <= length(yErr{2})
    if ( real(yErr{2}(k)) < yThr(1) ) && ( imag(yErr{2}(k)) >= yThr(2) )
        yErr{2}(k) = [];
    else
        k = k + 1;
    end
end

% yErr{3} holds all errors of y for x = -1 - j
yErr{3} = y{3};
k = 1;
while k <= length(yErr{3})
    if ( real(yErr{3}(k)) < yThr(1) ) && ( imag(yErr{3}(k)) < yThr(2) )
        yErr{3}(k) = [];
    else
        k = k + 1;
    end
end

% yErr{4} holds all errors of y for x = 1 - j
yErr{4} = y{4};
k = 1;
while k <= length(yErr{4})
    if ( real(yErr{4}(k)) >= yThr(1) ) && ( imag(yErr{4}(k)) < yThr(2) )
        yErr{4}(k) = [];
    else
        k = k + 1;
    end
end

% ----------Empirical Probability of receving an errors
yErrTotal = [
    yErr{1}
    yErr{2}
    yErr{3}
    yErr{4}
    ];
empProb_yErr = length(yErrTotal)/trials;
disp(['Empirical probability of dector error is ' num2str(empProb_yErr*100) '%'])

% --------------------Scatter Plot of Errors
subplot(122)
scatter(real(yErr{1}), imag(yErr{1}), 5, 'fill')
hold on
scatter(real(yErr{2}), imag(yErr{2}), 5, 'fill')
scatter(real(yErr{3}), imag(yErr{3}), 5, 'fill')
scatter(real(yErr{4}), imag(yErr{4}), 5, 'fill')
hold off
axis([y1Axis y2Axis])
xlabel('y_1'), ylabel('y_2')
title({
    'Received Errors'
    ['Gaussian Noise \lambda = ' num2str(lambda) ', \sigma^2 = ' num2str(1/(2*lambda))]
    ['Empirical Probability of Error = ' num2str(empProb_yErr*100) '%']
    })
grid on
legend([
    'y =  1 + j + n'
    'y = -1 + j + n'
    'y = -1 - j + n'
    'y =  1 - j + n'
    ])

% Decision Rule Threshold Lines
y1Thr = line('XData', [yThr(1) yThr(1)], 'YData', y2Axis);
y1Thr.LineWidth = 1;
y1Thr.Color = [0 0 0];
y1Thr.LineStyle = '--';
y1Thr.HandleVisibility = 'off';

y2Thr = line('XData', y1Axis, 'YData', [yThr(2) yThr(2)]);
y2Thr.LineWidth = 1;
y2Thr.Color = [0 0 0];
y2Thr.LineStyle = '--';
y2Thr.HandleVisibility = 'off';
%% Question 8d
% The effects of the gaussian noise parameter lambda on
% received signals
% Increasing lambda decreases the variance of the
% gaussian noise
clc, clear
% Implementation of Questions 8a, 8b, 8c are written in
% function rcvErr()

% Parameters
lambda = [
    0.5
    1
    5];
sigmaSq = 1./(2.*lambda);
% Probabilty of each possibility of X
p_ass2 = [0.25 0.25 0.25 0.25];
% Decision Rule Threshold
yThr_ass2 = [0 0];
% Number of Trials
trials = 10e3;

% Plot Axis
y1Axis = [-5 5];
y2Axis = [-5 5];

% --------------------Empirical Probability of Error
figure(4)
for k = 1:3
    [empProb_err(k,1), yRcv, yErr] = rcvErr(lambda(k), p_ass2, yThr_ass2, trials);

    % Scatter plot of all received signals
    subplot(320 + 2*k-1)
    plotRcv(yRcv, lambda(k), [y1Axis y2Axis])
    
    % Scatter Plot of Errors
    subplot(320 + 2*k)
    plotRcvErr(yErr, empProb_err(k), lambda(k), yThr_ass2, [y1Axis y2Axis])
    
end

% --------------------Theoretical Probability of Error
% Probability that X1 is decoded correctly AND X1 = -1
prob_X1crt = normcdf(1, 0, sqrt(sigmaSq)) * 0.5;
% Probability that X2 is decoded correctly AND X2 = -1
prob_X2crt = normcdf(1, 0, sqrt(sigmaSq)) * 0.5;
% Probability that X1 and X2 are decoded correctly AND
% X1 = -1 , X2 = 1, therefore X = -1-j
prob_X1nX2crt = prob_X1crt.*prob_X2crt;

% Probability that X is decoded correctly for all four
% cases of X = {1+j, -1+j, -1-j, 1-j} are the same
% because P[X = {1+j, -1+j, -1-j, 1-j}] are the same
% Total probability that X is decoded correctly
prob_Xcrt = prob_X1nX2crt*4;
% Total theorectical probability of error
theoProb_err = 1 - prob_Xcrt;

% --------------------------------------------------
% Tabulate, display, and save results for Assumption 2
ass2 = table(lambda, sigmaSq, empProb_err, theoProb_err);
disp('ASSUMPTION 2')
dispRcv(ass2, p_ass2, yThr_ass2)
save ass2.mat ass2 p_ass2 yThr_ass2

% ------------------------------Observations
% For lower values of lambda, the variance of the added
% gaussian noise will be higher.
% This causes the received signals Y to be more spread
% out, and therefore more of the received signals will
% lie outside the boundary of the decision rule
% thresholds. This then results in more detector
% errors.
% The converse is true for higher values of lambda,
% which is a lower variance of the gaussian noise.

%% Question 9
% Probabilities of the input X = {1+j, -1+j, -1-j, 1-j}
% are P_X = {3/4, 1/12, 1/12, 1/12} respectively
clc, clear

% Parameters
lambda = [
    0.5
    1
    5
    ];
sigmaSq = 1./(2.*lambda);
% Probabilty of each possibility of X
p_ass3 = [3/4 1/12 1/12 1/12];
% Decision Rule Threshold
yThr_ass3 = [0 0];
% Number of Trials
trials = 10e3;

% Plot Axis
y1y2axis = [-5 5 -5 5];

figure(5)
for k = 1:3
    [empProb_err(k,1), yRcv, yErr] = rcvErr(lambda(k), p_ass3, yThr_ass3, trials);
    
    % Scatter plot of all received signals
    subplot(330 + 3*k-2)
    plotRcv(yRcv, lambda(k), y1y2axis)
    
    % Scatter Plot of Errors
    subplot(330 + 3*k-1)
    plotRcvErr(yErr, empProb_err(k), lambda(k), yThr_ass3, y1y2axis)
   
end

% --------------------Theoretical Probability of Error
% Probability that X decoded correctly AND X = 1+j
% P[X = 1+j] = 3/4
% Probability of X decoded correctly in the first
% quadrant
prob_Xfirstcrt = normcdf(1,0,sqrt(sigmaSq)).*normcdf(1,0,sqrt(sigmaSq))*3/4;

% Probability that X decoded correctly AND X = -1-j
% P[X = -1-j] = 1/12
% Probability of X decoded correctly in the third
% quadrant
prob_Xthirdcrt = normcdf(1,0,sqrt(sigmaSq)).*normcdf(1,0,sqrt(sigmaSq))*1/12;
% Probability of X decoded correctly in the second,
% third, and fourth quadrants are the same because
% P[X = {-1+j, -1-j, 1-j}] are the same

% Total probability that X is decoded correctly
prob_Xcrt = prob_Xfirstcrt + prob_Xthirdcrt*3;
% Total theorectical probability of error
theoProb_err = 1 - prob_Xcrt;

% --------------------------------------------------
% Load Assumption 2 data from previous simulation
load ass2.mat
disp('ASSUMPTION 2')
dispRcv(ass2, p_ass2, yThr_ass2)

% --------------------------------------------------
% Tabulate, display, and save results for Assumption 3
ass3 = table(lambda, sigmaSq, empProb_err, theoProb_err);
% Display results for Assumption 3
disp('ASSUMPTION 3')
dispRcv(ass3, p_ass3, yThr_ass3)
% save ass3.mat ass3 p_ass3 yThr_ass3




% ------------------------------Observations
% Under assumption 3 where the probabilities of all
% possible transmitted x is {3/4, 1/12, 1/12, 1/12},
% more (blue) signal from transmitted x = 1 + j are
% present and the signals at the receiving end are also
% more spread out because the probabilty of x = 1 + j
% is the highest.

% This results in a higher (worse) overall detector
% error if the variance of the noise is large but also
% a lower (better) overall detector error if the
% variance of the noise is small.


% ----Adjust decision rule to improve detector error
% Adjust Decision Rule Threshold
yThr_ass3_adj = [-1 -1];
for k = 1:3
    [empProb_err(k,1), yRcv, yErr] = rcvErr(lambda(k), p_ass3, yThr_ass3_adj, trials);
    
    % Scatter Plot of Errors
    subplot(330 + 3*k)
    plotRcvErr(yErr, empProb_err(k), lambda(k), yThr_ass3_adj, y1y2axis)
   
end
ass3_adj = table(lambda, sigmaSq, empProb_err);

% % Save results of Assumption 3 with adjusted decision
% % rule
% save ass3_adj.mat ass3_adj p_ass3 yThr_ass3_adj

% Display results for Assumption 3 with adjusted decision
disp('ASSUMPTION 3 - with decision rule adjustments')
dispRcv(ass3_adj, p_ass3, yThr_ass3_adj)

% ------------------------------Observations
% With the decision rule adjusted to y1 = -1 and y2 = -1,
% the number of received errors with noise of higher
% variance is lowered (improved) but the number of
% received errors with noise of lower variance is
% gotten worse!

% This presents use with a trade-off.
% Adjusting the receiver's decision rule to filter out
% more errors will lower (improve) the detector error
% if the noise has a higher variance (noise with higher
% strength).
% Adjusting the receiver's decision rule to filter out
% more errors will increase (worsen) the detector error
% if the noise has a lower variance (noise with lower
% strength).

%% Question 10
% Probabilities of the input X = {1+j, -1+j, -1-j, 1-j}
% are P_X = {1, 0, 0, 0} respectively
clc, clear

% Parameters
lambda = [
    0.5
    1
    5
    ];
sigmaSq = 1./(2.*lambda);
% Probabilty of each possibility of X
p_ass4 = [1 0 0 0];
% Decision Rule Threshold
yThr_ass4 = [0 0];
% Number of Trials
trials = 10e3;

% Plot Axis
y1y2axis = [-5 5 -5 5];

figure(6)
for k = 1:3
    
    [empProb_err(k,1), yRcv, yErr] = rcvErr(lambda(k), p_ass4, yThr_ass4, trials);
    
    % Scatter plot of all received signals
    subplot(330 + 3*k-2)
    plotRcv(yRcv, lambda(k), y1y2axis)
    
    % Scatter Plot of Errors
    subplot(330 + 3*k-1)
    plotRcvErr(yErr, empProb_err(k), lambda(k), yThr_ass4, y1y2axis)
    
end
ass4 = table(lambda, sigmaSq, empProb_err);

% % Save results of Assumption 4
% save ass4.mat ass4 p_ass4 yThr_ass4

% Display results for Assumption 4
disp('ASSUMPTION 4')
dispRcv(ass4, p_ass4, yThr_ass4)


% ----Adjust decision rule to improve detector error
% Adjust Decision Rule Threshold
yThr_ass4_adj = [-4 -4];
for k = 1:3
    [empProb_err(k,1), yRcv, yErr] = rcvErr(lambda(k), p_ass4, yThr_ass4_adj, trials);
    
    % Scatter Plot of Errors
    subplot(330 + 3*k)
    plotRcvErr(yErr, empProb_err(k), lambda(k), yThr_ass4_adj, y1y2axis)
   
end
ass4_adj = table(lambda, sigmaSq, empProb_err);

% % Save results of Assumption 4 with adjusted decision
% % rule
% save ass4_adj.mat ass4_adj p_ass4 yThr_ass4_adj

% Display results for Assumption 3 with adjusted decision
disp('ASSUMPTION 4 - with decision rule adjustments')
dispRcv(ass4_adj, p_ass4, yThr_ass4_adj)

% ------------------------------Observations
% Using thresholds of y1 = -4 and y2 = -4 for the
% decision rule will yield zero detector error
% probability for all three values of lambda!

%% Plot Gaussian Noise
clc, clear

lambda = 1;

trials = 30e3;
for k = 1:trials
    [N1(k,1) N2(k,1) r] = gaussianNoise(lambda);
end

% N1 = normrnd(0, 1, [30e3, 1]);
% N2 = normrnd(0, 1, [30e3, 1]);

n = 1:length(N1);

figure(11)
subplot(221)
scatter(n, N1, 1, 'fill')
title('N_1')
ylim([-max(N1) max(N1)])

subplot(222)
scatter(n, N2, 1, 'fill')
title('N_2')
ylim([-max(N2) max(N2)])

subplot(223)
cdfplot(N1)

subplot(224)
cdfplot(N2)