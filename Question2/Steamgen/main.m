addpath(genpath('../../NNH'))
addpath(genpath('../../Common'))
load 'Steamgen.dat'
s = size(Steamgen);
idx=s(1);
SteamgenTrainSize = 6*s(1)/10;

SteamgenTestSize = 4*s(1)/10;

rand_idx = randperm(idx);

SteamgenTrain = Steamgen(rand_idx(SteamgenTestSize+1:s(1)), :);

SteamgenTest = Steamgen(rand_idx(1:SteamgenTestSize), :);

SteamgenTrainX = SteamgenTrain(:,[1,2,3,4])';
SteamgenTrainY1 = SteamgenTrain(:,5)';
SteamgenTrainY2 = SteamgenTrain(:,6)';
SteamgenTrainY3 = SteamgenTrain(:,7)';
SteamgenTrainY4 = SteamgenTrain(:,8)';

SteamgenTestX = SteamgenTest(:,[1,2,3,4])';
SteamgenTestY1 = SteamgenTest(:,5)';
SteamgenTestY2 = SteamgenTest(:,6)';
SteamgenTestY3 = SteamgenTest(:,7)';
SteamgenTestY4 = SteamgenTest(:,8)';

% trainlm
tic
[mse_lm_steamgen_y1, out_lm_steamgen_y1] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY1, SteamgenTestX, SteamgenTestY1, 'trainlm');
time_lm_steamgen_y1 = toc;

tic
[mse_lm_steamgen_y2, out_lm_steamgen_y2] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY2, SteamgenTestX, SteamgenTestY2, 'trainlm');
time_lm_steamgen_y2 = toc;

tic
[mse_lm_steamgen_y3, out_lm_steamgen_y3] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY3, SteamgenTestX, SteamgenTestY3, 'trainlm');
time_lm_steamgen_y3 = toc;

tic
[mse_lm_steamgen_y4, out_lm_steamgen_y4] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY4, SteamgenTestX, SteamgenTestY4, 'trainlm');
time_lm_steamgen_y4 = toc;

% traingd
tic
[mse_gd_steamgen_y1, out_gd_steamgen_y1] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY1, SteamgenTestX, SteamgenTestY1, 'traingd');
time_gd_steamgen_y1 = toc;

tic
[mse_gd_steamgen_y2, out_gd_steamgen_y2] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY2, SteamgenTestX, SteamgenTestY2, 'traingd');
time_gd_steamgen_y2 = toc;

tic
[mse_gd_steamgen_y3, out_gd_steamgen_y3] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY3, SteamgenTestX, SteamgenTestY3, 'traingd');
time_gd_steamgen_y3 = toc;

tic
[mse_gd_steamgen_y4, out_gd_steamgen_y4] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY4, SteamgenTestX, SteamgenTestY4, 'traingd');
time_gd_steamgen_y4 = toc;

% traingdm
tic
[mse_gdm_steamgen_y1, out_gdm_steamgen_y1] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY1, SteamgenTestX, SteamgenTestY1, 'traingdm');
time_gdm_steamgen_y1 = toc;

tic
[mse_gdm_steamgen_y2, out_gdm_steamgen_y2] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY2, SteamgenTestX, SteamgenTestY2, 'traingdm');
time_gdm_steamgen_y2 = toc;

tic
[mse_gdm_steamgen_y3, out_gdm_steamgen_y3] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY3, SteamgenTestX, SteamgenTestY3, 'traingdm');
time_gdm_steamgen_y3 = toc;

tic
[mse_gdm_steamgen_y4, out_gdm_steamgen_y4] = mlpTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY4, SteamgenTestX, SteamgenTestY4, 'traingdm');
time_gdm_steamgen_y4 = toc;


% rb
tic
[mse_rb_steamgen_y1, out_rb_steamgen_y1] = rbTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY1, SteamgenTestX, SteamgenTestY1, 0.001, 1, 5);
time_rb_steamgen_y1 = toc;

tic
[mse_rb_steamgen_y2, out_rb_steamgen_y2] = rbTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY2, SteamgenTestX, SteamgenTestY2, 0.001, 1, 5);
time_rb_steamgen_y2 = toc;

tic
[mse_rb_steamgen_y3, out_rb_steamgen_y3] = rbTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY3, SteamgenTestX, SteamgenTestY3, 0.001, 1, 5);
time_rb_steamgen_y3 = toc;

tic
[mse_rb_steamgen_y4, out_rb_steamgen_y4] = rbTrainingFunction(2000, SteamgenTrainX, SteamgenTrainY4, SteamgenTestX, SteamgenTestY4, 0.001, 1, 5);
time_rb_steamgen_y4 = toc;


% rbe
% because of low memory; one of each 10 items is selected!
SteamgenTrainSize = 6*s(1)/100;

SteamgenTestSize = 4*s(1)/100;

SteamgenTrain = Steamgen(rand_idx(SteamgenTestSize+1:SteamgenTestSize+SteamgenTrainSize), :);
SteamgenTrainMean = mean(Steamgen);
SteamgenTrainMin = min(Steamgen);

for i=1:size(SteamgenTrain, 2)
    SteamgenTrain(:,i) = (SteamgenTrain(:,i) - SteamgenTrainMin(i))/SteamgenTrainMean(i);
end

SteamgenTest = Steamgen(rand_idx(1:SteamgenTestSize), :);

for i=1:size(SteamgenTest, 2)
    SteamgenTest(:,i) = (SteamgenTest(:,i) - SteamgenTrainMin(i))/SteamgenTrainMean(i);
end


SteamgenTrainX = SteamgenTrain(:,[1,2,3,4])';
SteamgenTrainY1 = SteamgenTrain(:,5)';
SteamgenTrainY2 = SteamgenTrain(:,6)';
SteamgenTrainY3 = SteamgenTrain(:,7)';
SteamgenTrainY4 = SteamgenTrain(:,8)';

SteamgenTestX = SteamgenTest(:,[1,2,3,4])';
SteamgenTestY1 = SteamgenTest(:,5)';
SteamgenTestY2 = SteamgenTest(:,6)';
SteamgenTestY3 = SteamgenTest(:,7)';
SteamgenTestY4 = SteamgenTest(:,8)';

tic
[mse_rbe_steamgen_y1, out_rbe_steamgen_y1] = rbeTrainingFunction(10, SteamgenTrainX, SteamgenTrainY1, SteamgenTestX, SteamgenTestY1, 1);
time_rbe_steamgen_y1 = toc;

tic
[mse_rbe_steamgen_y2, out_rbe_steamgen_y2] = rbeTrainingFunction(10, SteamgenTrainX, SteamgenTrainY2, SteamgenTestX, SteamgenTestY2, 1);
time_rbe_steamgen_y2 = toc;

tic
[mse_rbe_steamgen_y3, out_rbe_steamgen_y3] = rbeTrainingFunction(10, SteamgenTrainX, SteamgenTrainY3, SteamgenTestX, SteamgenTestY3, 1);
time_rbe_steamgen_y3 = toc;

tic
[mse_rbe_steamgen_y4, out_rbe_steamgen_y4] = rbeTrainingFunction(10, SteamgenTrainX, SteamgenTrainY4, SteamgenTestX, SteamgenTestY4, 1);
time_rbe_steamgen_y4 = toc;