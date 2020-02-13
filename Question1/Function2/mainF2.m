train_test_F2
addpath(genpath('../../Common'));
[trainXF2, trainYF2, testXF2, testYF2] = loadDataF2();

% trainlm
tic
[mse_lm_f2, out_lm_f2] = mlpTrainingFunction(1000, trainXF2, trainYF2, testXF2, testYF2, 'trainlm');
time_lm_f2 = toc;

% traingd
tic
[mse_gd_f2, out_gd_f2] = mlpTrainingFunction(1000, trainXF2, trainYF2, testXF2, testYF2, 'traingd');
time_gd_f2 = toc;

% traingdm
tic
[mse_gdm_f2, out_gdm_f2] = mlpTrainingFunction(1000, trainXF2, trainYF2, testXF2, testYF2, 'traingdm');
time_gdm_f2 = toc;

% rb
tic
[mse_rb_f2, out_rb_f2] = rbTrainingFunction(1000, trainXF2, trainYF2, testXF2, testYF2, 0.001, 1, 5);
time_rb_f2 = toc;

% rbe
tic
[mse_rbe_f2, out_rbe_f2] = rbeTrainingFunction(1000, trainXF2, trainYF2, testXF2, testYF2, 1);
time_rbe_f2 = toc;
