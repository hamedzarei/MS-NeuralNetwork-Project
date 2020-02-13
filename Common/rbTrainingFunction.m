function  [o_mse, testOutput] = rbTrainingFunction( epochs, trainX, trainY, testX, testY, goal, spread, neuron )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net_rb = newrb(trainX, trainY, goal, spread, neuron);
net_rb.trainFcn = 'trainlm';
net_rb.trainParam.epochs = epochs;

% https://www.mathworks.com/matlabcentral/answers/377267-how-to-predict-from-a-trained-neural-network
mse_numbers = [];
for i = 1:5
   trainedNet = train(net_rb, trainX, trainY);
   testOutput = trainedNet(testX);
   diff = testY - testOutput;
   mse = mean(diff.^2);
   mse_numbers = [mse_numbers; mse];
end
o_mse = mse_numbers;

end

