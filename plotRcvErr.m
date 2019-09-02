% plotRcv plots a scatter plot of received signals in
% four quadrants

% lambda is the parameter of the Exponential RV of R^2
% of the Gaussian Noise
% lambda = 1/(2*sigma^2) where sigma^2 is the variance
% of the Gaussian Noise

% yThres = [y1 y2] is the decision rule thresholds on
% the y1 and y2 axis respectively

function plotRcvErr(y, empProb_yErr, lambda, yThr, y1y2Axis)
    scatter(real(y{1}), imag(y{1}), 5, 'fill')
    hold on
    scatter(real(y{2}), imag(y{2}), 5, 'fill')
    scatter(real(y{3}), imag(y{3}), 5, 'fill')
    scatter(real(y{4}), imag(y{4}), 5, 'fill')
    hold off
    xlabel('y_1'), ylabel('y_2')
    title({
        'Received Errors'
        ['Gaussian Noise \lambda = ' num2str(lambda) ', \sigma^2 = ' num2str(1/(2*lambda))]
        ['Empirical Probability of Error = ' num2str(empProb_yErr*100) '%']
        })
    grid on
    legend(['y =  1 + j + n'
        'y = -1 + j + n'
        'y = -1 - j + n'
        'y =  1 - j + n'
        ])
    axis(y1y2Axis)
    
    % Decision Rule Threshold Lines
    y1Thr = line('XData', [yThr(1) yThr(1)], 'YData', y1y2Axis(3:4));
    y1Thr.LineWidth = 1;
    y1Thr.Color = [0 0 0];
    y1Thr.LineStyle = '--';
    y1Thr.HandleVisibility = 'off';
    
    y2Thr = line('XData', y1y2Axis(1:2), 'YData', [yThr(2) yThr(2)]);
    y2Thr.LineWidth = 1;
    y2Thr.Color = [0 0 0];
    y2Thr.LineStyle = '--';
    y2Thr.HandleVisibility = 'off';
end