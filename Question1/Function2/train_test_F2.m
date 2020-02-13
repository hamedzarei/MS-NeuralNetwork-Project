myF2Min = 1;
myF2Max = 6;
myF2TrainSize = 216;
myF2X = myF2Min + (myF2Max-myF2Min).*rand(myF2TrainSize,1);
myF2Y = myF2Min + (myF2Max-myF2Min).*rand(myF2TrainSize,1);
myF2Z = myF2Min + (myF2Max-myF2Min).*rand(myF2TrainSize,1);
myF2OutTrain = [];
for row = 1:myF2TrainSize
    myF2OutTrain = [myF2OutTrain;myF2(myF2X(row), myF2Y(row), myF2Z(row))];
end
myF2OutTrainFile = [myF2X myF2Y myF2Z myF2OutTrain];
dlmwrite('myF2OutTrainFile.dat',myF2OutTrainFile, '\t');

myF2TestSize = 125;
myF2X = myF2Min + (myF2Max-myF2Min).*rand(myF2TestSize,1);
myF2Y = myF2Min + (myF2Max-myF2Min).*rand(myF2TestSize,1);
myF2Z = myF2Min + (myF2Max-myF2Min).*rand(myF2TestSize,1);
myF2OutTest = [];
for row = 1:myF2TestSize
    myF2OutTest = [myF2OutTest;myF2(myF2X(row), myF2Y(row), myF2Z(row))];
end
myF2OutTestFile = [myF2X myF2Y myF2Z myF2OutTest];
dlmwrite('myF2OutTestFile.dat',myF2OutTestFile, '\t');