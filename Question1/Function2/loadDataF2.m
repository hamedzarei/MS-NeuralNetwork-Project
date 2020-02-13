function [ trainX, trainY, testX, testY ] = loadDataF2()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load myF2OutTrainFile.dat;
trainX = myF2OutTrainFile(:, [1,2,3])';
trainY = myF2OutTrainFile(:, 4)';
load myF2OutTestFile.dat;
testX = myF2OutTestFile(:, [1,2,3])';
testY = myF2OutTestFile(:, 4)';

end

