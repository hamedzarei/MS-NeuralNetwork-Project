addpath(genpath('../../NNH'))
addpath(genpath('../../Common'))
load 'CDPlayer.dat'
s = size(CDPlayer);
idx=s(1);
CDPlayerTrainSize = 6*s(1)/10;

CDPlayerTestSize = 4*s(1)/10;

rand_idx = randperm(idx);

CDPlayerTrain = CDPlayer(rand_idx(CDPlayerTestSize+1:s(1)), :);

CDPlayerTest = CDPlayer(rand_idx(1:CDPlayerTestSize), :);

CDPlayerTrainX = CDPlayerTrain(:,[1,2])';
CDPlayerTrainY1 = CDPlayerTrain(:,3)';
CDPlayerTrainY2 = CDPlayerTrain(:,4)';

CDPlayerTestX = CDPlayerTest(:,[1,2])';
CDPlayerTestY1 = CDPlayerTest(:,3)';
CDPlayerTestY2 = CDPlayerTest(:,4)';


% trainlm
tic
[mse_lm_cdplayer_y1, out_lm_cdplayer_y1] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY1, CDPlayerTestX, CDPlayerTestY1, 'trainlm');
time_lm_cdplayer_y1 = toc;

tic
[mse_lm_cdplayer_y2, out_lm_cdplayer_y2] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY2, CDPlayerTestX, CDPlayerTestY2, 'trainlm');
time_lm_cdplayer_y2 = toc;


% traingd
tic
[mse_gd_cdplayer_y1, out_gd_cdplayer_y1] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY1, CDPlayerTestX, CDPlayerTestY1, 'traingd');
time_gd_cdplayer_y1 = toc;

tic
[mse_gd_cdplayer_y2, out_gd_cdplayer_y2] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY2, CDPlayerTestX, CDPlayerTestY2, 'traingd');
time_gd_cdplayer_y2 = toc;

% traingdm
tic
[mse_gdm_cdplayer_y1, out_gdm_cdplayer_y1] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY1, CDPlayerTestX, CDPlayerTestY1, 'traingdm');
time_gdm_cdplayer_y1 = toc;

tic
[mse_gdm_cdplayer_y2, out_gdm_cdplayer_y2] = mlpTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY2, CDPlayerTestX, CDPlayerTestY2, 'traingdm');
time_gdm_cdplayer_y2 = toc;

% rb
tic
[mse_rb_cdplayer_y1, out_rb_cdplayer_y1] = rbTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY1, CDPlayerTestX, CDPlayerTestY1, 0.001, 1, 5);
time_rb_cdplayer_y1 = toc;

tic
[mse_rb_cdplayer_y2, out_rb_cdplayer_y2] = rbTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY2, CDPlayerTestX, CDPlayerTestY2, 0.001, 1, 5);
time_rb_cdplayer_y2 = toc;


% rbe
% because of low memory; one of each 10 items is selected!
CDPlayerTrainSize = 6*s(1)/100;

CDPlayerTestSize = 4*s(1)/100;

CDPlayerTrain = CDPlayer(rand_idx(CDPlayerTestSize+1:CDPlayerTestSize+CDPlayerTrainSize), :);
CDPlayerTrainMean = mean(CDPlayer);
CDPlayerTrainMin = min(CDPlayer);

for i=1:size(CDPlayerTrain, 2)
    CDPlayerTrain(:,i) = (CDPlayerTrain(:,i) - CDPlayerTrainMin(i))/CDPlayerTrainMean(i);
end

CDPlayerTest = CDPlayer(rand_idx(1:CDPlayerTestSize), :);

for i=1:size(CDPlayerTest, 2)
    CDPlayerTest(:,i) = (CDPlayerTest(:,i) - CDPlayerTrainMin(i))/CDPlayerTrainMean(i);
end


CDPlayerTrainX = CDPlayerTrain(:,[1,2])';
CDPlayerTrainY1 = CDPlayerTrain(:,3)';
CDPlayerTrainY2 = CDPlayerTrain(:,4)';

CDPlayerTestX = CDPlayerTest(:,[1,2])';
CDPlayerTestY1 = CDPlayerTest(:,3)';
CDPlayerTestY2 = CDPlayerTest(:,4)';

tic
[mse_rbe_cdplayer_y1, out_rbe_cdplayer_y1] = rbeTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY1, CDPlayerTestX, CDPlayerTestY1, 1);
time_rbe_cdplayer_y1 = toc;

tic
[mse_rbe_cdplayer_y2, out_rbe_cdplayer_y2] = rbeTrainingFunction(2000, CDPlayerTrainX, CDPlayerTrainY2, CDPlayerTestX, CDPlayerTestY2, 1);
time_rbe_cdplayer_y2 = toc;
