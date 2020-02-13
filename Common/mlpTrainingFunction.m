function  [o_mse, testOutput] = mlpTrainingFunction( epochs, trainX, trainY, testX, testY, trainFunc )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
net_mlp_gdm = newff(trainX, trainY, {5, 5}, {'tansig', 'tansig', 'purelin'}, trainFunc);
net_mlp_gdm.trainParam.epochs = epochs;
% https://www.mathworks.com/matlabcentral/answers/377267-how-to-predict-from-a-trained-neural-network
mse_numbers = [];
for i = 1:5
   trainedNet = train(net_mlp_gdm, trainX, trainY);
   testOutput = trainedNet(testX);
   diff = testY - testOutput;
   mse = mean(diff.^2);
   mse_numbers = [mse_numbers; mse];
end
o_mse = mse_numbers;

end

