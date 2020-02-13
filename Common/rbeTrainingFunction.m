function  [o_mse, testOutput] = rbeTrainingFunction( epochs, trainX, trainY, testX, testY, spread )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net_rbe = newrbe(trainX, trainY, spread);
net_rbe.trainFcn = 'trainlm';
net_rbe.trainParam.epochs = epochs;

% https://www.mathworks.com/matlabcentral/answers/377267-how-to-predict-from-a-trained-neural-network
mse_numbers = [];
for i = 1:5
   trainedNet = train(net_rbe, trainX, trainY);
   testOutput = trainedNet(testX);
   diff = testY - testOutput;
   mse = mean(diff.^2);
   mse_numbers = [mse_numbers; mse];
end
o_mse = mse_numbers;

end

