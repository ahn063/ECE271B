% Andy Nguyen
% 2/13/2024
% ECE 271B

%% Setup
close all; clc;
if ~(exist('imgs', 'var') && exist('labels', 'var') && exist('imgs_test', 'var') && exist('labels_test', 'var'))  
    [imgs, labels] = readMNIST('training set/train-images-idx3-ubyte', 'training set/train-labels-idx1-ubyte', 60000, 0);
    [imgs_test, labels_test] = readMNIST('test set/t10k-images-idx3-ubyte', 'test set/t10k-labels-idx1-ubyte', 10000, 0);
end

%% Problem 5a

% hotBitLabel = hotBit(labels, 10);
d = size(imgs, 2); %dimension of vectors d = 28 x 28 = 784
k = 10; % number of classes
if ~(exist('w', 'var') && exist('iterNum', 'var') && exist('costVector', 'var'))
    w = randn(d, k); % randn intialization of weights of mean 0, unit variance
    iterNum = 0;
    %init temp variables to start training loop
    costVector(2) = 2;
    costVector(1) = 1;
    cost = costVector(1);
end

tol = 1e-4;
lr = 1e-5; %learning rate
totalIter = 1000;

%train until convergence
while(abs(costVector(end) - costVector(end-1)) > tol)
    [w, cost, y] = one_layer(imgs, labels, w, k, lr);
    iterNum = iterNum+1;
    costVector(iterNum) = cost;
    training_error(iterNum) = calcErrorSingle(imgs, labels, w);
    test_error(iterNum) = calcErrorSingle(imgs_test, labels_test, w);
    %output
    if (mod(iterNum, 1000) == 0)
        iterNum
        cost
        training_error(end)
        test_error(end)
    end
end

figure()
plot(training_error)
hold on;
grid on;
plot(test_error)
title('Single Layer: Error vs. number of iterations')
legend('Training Error', 'Test Error')
xlabel('Iterations')
ylabel('Error %')

figure()
plot(costVector)
hold on;
grid on;
title('Single Layer: Cross Entropy Loss vs. number of iterations')
legend('Cost')
xlabel('Iterations')
ylabel('Cross Entropy Loss')


%% Part 5b
%sigmoid network
tol = 50e-5;
maxIter = 10e4;
k = 10; %numClasses
lr = 10e-5; %learning rate
H = 10; % number of hidden layers
tStart = tic;
[~, trainPercentError, testPercentError] = two_layer_sigmoid(imgs, labels, imgs_test, labels_test, k, lr, 10, tol, maxIter);

figure()
plot(trainPercentError)
hold on;
plot(testPercentError)
grid on;
title('Two Layer Sigmoid: Training and Test Error H = 10')
ylabel('Error %')
xlabel('Iterations')

[~, trainPercentError, testPercentError] = two_layer_sigmoid(imgs, labels, imgs_test, labels_test, k, lr, 20, tol, maxIter);

figure()
plot(trainPercentError)
hold on;
plot(testPercentError)
grid on;
title('Two Layer Sigmoid: Training and Test Error H = 20')
ylabel('Error %')
xlabel('Iterations')

[~, trainPercentError, testPercentError] = two_layer_sigmoid(imgs, labels, imgs_test, labels_test, k, lr, 50, tol, maxIter);

figure()
plot(trainPercentError)
hold on;
plot(testPercentError)
grid on;
title('Two Layer Sigmoid: Training and Test Error H = 50')
ylabel('Error %')
xlabel('Iterations')

elapsedTime = toc(tStart)


%% init values

%% Two Layer ReLU Network



%% Stochastic Grad Descent (mini batch)


%%

function [losses, trainPercentError, testPercentError] = two_layer_sigmoid(imgs, labels, imgs_test, labels_test, k, lr, H, tol, maxIter)
    n = size(imgs, 1); %num of samples
    batchSize = n; %use all examples in batch training
    d = size(imgs, 2);
    labelHotBit = hotBit(labels, k); %hot bit encoded labels
    hiddenWeights = [randn(H, d), zeros(H, 1)]; %first layer weights + bias
    outputWeights = [randn(k, H), zeros(k, 1)]; %second layer weights + bias
    iterNum = 1;
    losses = [1, 2];
    while (abs(losses(end) - losses(end-1)) > tol && iterNum < maxIter)
        %forward pass
        X = [imgs ,ones(n, 1)]; %60000 x 785
        g = hiddenWeights*X';  %10 x 60000
        y = [sigmoid(g); ones(1, size(g,2))]; %11 x 60000
        u = outputWeights * y; %10 x 60000
        z = softmax(u'); %60000 x 10; transposed u to fit softmax function (row wise averaging)
        
        %backward pass
        % Error at output layer
        error_output = z - labelHotBit; % 60000 x 10
        
        % Gradient for output weights
        d_outputWeights = error_output' * y'; % 10 x (H+1)
        
        % Calculate sigmoid derivative for hidden layer outputs
        % Exclude the bias row for the derivative calculation
        sigmoid_derivative = y(1:end-1, :) .* (1 - y(1:end-1, :)); % H x 60000
        
        % Error at hidden layer (backpropagate through the output weights)
        % Include the bias weights in the backpropagation
        error_hidden = (outputWeights' * error_output') .* [sigmoid_derivative; ones(1, size(sigmoid_derivative, 2))]; % (H+1) x 60000
        
        % Gradient for hidden weights
        % Since X already includes a bias term, we directly use it for calculating the gradient
        d_hiddenWeights = error_hidden(1:end-1, :) * X; % H x (d+1)
        
        % Update weights
        outputWeights = outputWeights - lr * d_outputWeights; % Update output weights directly
        hiddenWeights = hiddenWeights - lr * d_hiddenWeights; % Update hidden weights directly
        
        losses(iterNum) = -sum(labelHotBit.*log(z), 'all') / n;
        trainPercentError(iterNum) = calcErrorDouble(labels, z);
        testPercentError(iterNum) = calcErrorDouble(labels_test, forward_pass_all(imgs_test, hiddenWeights, outputWeights));
        iterNum = iterNum + 1;
    end
end

function [w, cost, y] = one_layer(imgs, labels, w, numClasses, learningRate)
    a = imgs*w; %should be n x k
    y = softmax(a); % n x k, calculates score of each example
    t = hotBit(labels, numClasses);
    cost = crossEntropyCost(t,y);
    delta_w = backprop(t, y, imgs);
    w = w + learningRate*delta_w; % d x k
end

function output = d_sigmoid(u)
    output = sigmoid(u).*(1-sigmoid(u));
end

function output = sigmoid(u)
    output = 1./(1+exp(-u));
end

function output = ReLU(u)
    %u is vector
    output = max(max(0, u));
end

function output = softmax(a)
    exp_a = exp(a - max(a, [], 2)); % Improve numerical stability

    output = exp_a ./ sum(exp_a, 2);
end

function hotBitVector = hotBit(v, numClasses)
    %v is n x 1, where n = numExamples
    n = size(v,1);
    hotBitVector = zeros(n, numClasses);
    for i = 1:n
        hotBitVector(i, v(i)+1) = 1;
    end
end

function cost = crossEntropyCost(t, y)
    %t is hot bit encoded labels, n x k
    % y is n x k
    cost = -sum(sum(t.*log(y))) / size(t,1);
end

function delta_w = backprop(t, y, x)
    delta_w = x'*(t-y);
end

function percentError = calcErrorSingle(imgs, labels, w)
    totalN = size(imgs, 1);
    y = softmax(imgs*w);
    [~, predicted_labels] = max(y, [], 2);
    predicted_labels = predicted_labels - 1;
    percentCorrect = sum(predicted_labels==labels) / totalN;
    percentError = 1-percentCorrect;
end

function percentError = calcErrorDouble(labels, z)
    totalN = size(labels,1);
    [~, predicted_labels] = max(z, [], 2);
    predicted_labels = predicted_labels - 1;
    percentCorrect = sum(predicted_labels==labels) / totalN;
    percentError = 1-percentCorrect;
end

function cost = crossEntropyCostDouble(t, y)
    % Assuming t and y are both matrices of the same size
    cost = -sum(sum(t .* log(y))) / size(t, 1);
end

function z = forward_pass_all(imgs, w1, w2)
    % Forward pass for all samples, used to compute the cost
    % assumed included biases in w1 and w2 already
    X = [imgs ,ones(size(imgs,1), 1)]; %60000 x 785
    g = w1*X';  %10 x 60000
    y = [sigmoid(g); ones(1, size(g,2))]; %11 x 60000
    u = w2 * y; %10 x 60000
    z = softmax(u'); %60000 x 10; transposed u to fit softmax function (row wise averaging)
end