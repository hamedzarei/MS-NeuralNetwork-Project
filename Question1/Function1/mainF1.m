addpath(genpath('../../Common'));
train_test_F1
[trainXF1, trainYF1, testXF1, testYF1] = loadDataF1();

% trainlm
tic
[mse_lm_f1, out_lm_f1] = mlpTrainingFunction(1000, trainXF1, trainYF1, testXF1, testYF1, 'trainlm');
time_lm_f1 = toc;

% traingd
tic
[mse_gd_f1, out_gd_f1] = mlpTrainingFunction(1000, trainXF1, trainYF1, testXF1, testYF1, 'traingd');
time_gd_f1 = toc;

% traingdm
tic
[mse_gdm_f1, out_gdm_f1] = mlpTrainingFunction(1000, trainXF1, trainYF1, testXF1, testYF1, 'traingdm');
time_gdm_f1 = toc;

% rb
tic
[mse_rb_f1, out_rb_f1] = rbTrainingFunction(1000, trainXF1, trainYF1, testXF1, testYF1, 0.001, 1, 5);
time_rb_f1 = toc;

% rbe
tic
[mse_rbe_f1, out_rbe_f1] = rbeTrainingFunction(1000, trainXF1, trainYF1, testXF1, testYF1, 1);
time_rbe_f1 = toc;
