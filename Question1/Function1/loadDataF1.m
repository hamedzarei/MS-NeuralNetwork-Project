function [ trainX, trainY, testX, testY ] = loadDataF1()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load myF1OutTrainFile.dat;
trainX = myF1OutTrainFile(:, [1,2])';
trainY = myF1OutTrainFile(:, 3)';
load myF1OutTestFile.dat;
testX = myF1OutTestFile(:, [1,2])';
testY = myF1OutTestFile(:, 3)';

end

