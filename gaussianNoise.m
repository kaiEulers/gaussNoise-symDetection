% Noise Generator
function [n1 n2 r] = gaussianNoise(lambda)
    
    % Create RV of rSq and theta
    rSq = -1/lambda * log(1 - rand());
    theta = 2*pi*rand();
    
    % Transform into r, n1, n2
    r = sqrt(rSq);
    n1 = r*cos(theta);
    n2 = r*sin(theta);
    
end