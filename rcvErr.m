% lambda is the parameter of the Exponential RV of R^2
% of the Gaussian Noise
% lambda = 1/(2*sigma^2) where sigma^2 is the variance
% of the Gaussian Noise

% p(1), p(2), p(3), p(4) are the probabilities of the
% input X = {1+j, -1+j, -1-j, 1-j} respectively

% yThres = [y1 y2] is the decision rule thresholds on
% the y1 and y2 axis respectively

% trials is the number of trials the simulation will
% utilise

function [prob_yErr, y, yErr] = rcvErr(lambda, p, yThr, trials)

if (length(p) ~= 4) || (sum(p) ~= 1)
    % Error checking
    disp('Error!')
    disp('The sum of all four probabilities must equal to 1.')
    disp('Please try again.')
else
    % X = {1+j, -1+j, -1-j, 1-j}
    % N = Gaussian(0, 0.5) + j*Gaussian(0, 0.5)
    
    for k = 1:4
       y{k} = []; 
       yErr{k} = [];
    end
    
    % Vector count variables for all four possibility
    % of y
    k1 = 1; k2 = 1; k3 = 1; k4 = 1;
    for k0 = 1:trials
        % Generate Noise
        [n1 n2] = gaussianNoise(lambda);
        n = n1 + j*n2;
        
        % Generate random vector x = x1 + j*x2
        die = rand();
        cnt = 1;
        if die <= p(1)
            % x = 1 + j, 1st quandrant
            x = 1 + j;
            y{1}(k1,1) = x + n;
            k1 = k1 + 1;
        elseif (die > p(1)) && (die <= (p(1)+p(2)))
            % x = -1 + j, 2nd quandrant
            x = -1 + j;
            y{2}(k2,1) = x + n;
            k2 = k2 + 1;
        elseif (die > (p(1)+p(2))) && (die <= (p(1)+p(2)+p(3)))
            % x = -1 - j, 3rd quandrant
            x = -1 - j;
            y{3}(k3,1) = x + n;
            k3 = k3 + 1;
        else
            % x = 1 - j, 4th quandrant
            x = 1 - j;
            y{4}(k4,1) = x + n;
            k4 = k4 + 1;
        end
        cnt = cnt + 1;
    end
    
    % ---------------------------------------------
    % Remove correctly decoded Ys with decision rule
    % yErr{1} holds all errors of y for x = 1 + j
    yErr{1} = y{1};
    k = 1;
    while k <= length(yErr{1})
        if ( real(yErr{1}(k)) >= yThr(1) ) && ( imag(yErr{1}(k)) >= yThr(2) )
            % If y is decoded correctly, remove from
            % the vector
            yErr{1}(k) = [];
        else
            % Otherwise, move to the next element of
            % the vector
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
    
    % ----Empirical Probability of receving an errors
    yErrTotal = [
        yErr{1}
        yErr{2}
        yErr{3}
        yErr{4}
        ];
    prob_yErr = length(yErrTotal)/trials;
end