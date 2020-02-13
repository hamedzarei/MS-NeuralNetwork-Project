
function [] = jordan(trainX, trainY, testX)
feature_length=size(trainX, 2);  % selecting top 5 most significant genes
train_data=trainX';
test_data=testX';
class_train = trainY;
N1=5;                  % Middle Layer Neurons% Middle Layer Neurons
N2=1;                   % Output Layer Neurons% Output Layer Neurons
N0=feature_length+1;    % Input Layer Neurons (feature length + bias)% Input Layer Neurons (feat 

% Training parameters
eta = 0.5; % Learning Rate 
epoch=10;  % Training iterations

% Initialization of weights
w1=randn(N1,N0);    % Initial weights of input and middle layer connections (5, 3)
w2=randn(N2,N1);    % Initial weights of middle and output layer connections (1, 5)
wR=randn(N1, 1);
xR = 0;
for j=1:epoch
    % randomization of training data improves learning performance
    ind(j,:)=randperm(length(class_train));

    for k=1:size(train_data,1)

        Input=[1 train_data(ind(j,k),:)];  % {1} is for bias

        % Input layer
        n1 = w1*Input' + wR*xR;
        a1=tansig(n1);

        % Hidden layer
        n2 = w2*a1;
        a2=purelin(n2);

        % output layer
        Output(k)=a2;
        e = class_train(ind(j,k)) - Output(k);

        % Training of NN using Backpropagation learning algorithm (gradient
        % descent-based learning rule)

        df2 = 2*1*e;  % local gradient of Output Layer
        df1 = diag(dtansig(n1,a1),0)*w2'*df2; % local gradient of Hidden Layer
        %dxR = df1*xR;
        w1 = w1 + eta*df1*Input;  % input layer neurons weight update
        w2 = w2 + eta*df2*a1';    % hidden layer neurons weight update
        wR = wR + eta*df1*(xR);
        xR = a2;
        SE(j,k)= e*e';      % squared error
    end
    MSE(j)=mean(SE(j,:));       % objective function (mean squared error)

end