% data is the tabulated data
% p = {p1, p2, p3, p4} are the probabilities of each quadrant
% yThr = [y1 y2] are the decision rule thresholds

function dispRcv(data, p, yThr)
    
    disp(['Px = {' num2str(p(1)) ', ' num2str(p(2)) ', ' num2str(p(3)) ', ' num2str(p(4)) '}'])
    disp(['Desicion Rule Thresholds y1 = ' num2str(yThr(1)) ' y2 = ' num2str(yThr(2))])
    disp(' ')
    disp(data)
    disp('--------------------------------------------------')
    disp(' ')
    
end