myF1Min = -10;
myF1Max = 10;
myF1TrainSize = 121;
myF1X = myF1Min + (myF1Max-myF1Min).*rand(myF1TrainSize,1);
myF1Y = myF1Min + (myF1Max-myF1Min).*rand(myF1TrainSize,1);
myF1OutTrain = [];
for row = 1:myF1TrainSize
    myF1OutTrain = [myF1OutTrain;myF1(myF1X(row), myF1Y(row))];
end
myF1OutTrainFile = [myF1X myF1Y myF1OutTrain];
dlmwrite('myF1OutTrainFile.dat',myF1OutTrainFile, '\t');

myF1TestSize = 71;
myF1X = myF1Min + (myF1Max-myF1Min).*rand(myF1TestSize,1);
myF1Y = myF1Min + (myF1Max-myF1Min).*rand(myF1TestSize,1);
myF1OutTest = [];
for row = 1:myF1TestSize
    myF1OutTest = [myF1OutTest;myF1(myF1X(row), myF1Y(row))];
end
myF1OutTestFile = [myF1X myF1Y myF1OutTest];
dlmwrite('myF1OutTestFile.dat',myF1OutTestFile, '\t');