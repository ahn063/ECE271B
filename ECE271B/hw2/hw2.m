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
title('Error vs. number of iterations')
legend('Training Error', 'Test Error')
xlabel('Iterations')
ylabel('Error %')

figure()
plot(costVector)
hold on;
grid on;
title('Cross Entropy Loss vs. number of iterations')
legend('Cost')
xlabel('Iterations')
ylabel('Cross Entropy Loss')


%% Part 5b
n = size(imgs, 1); %num of samples
d = size(imgs, 2);
k = 10; %numClasses
lr = 10e-5; %learning rate
batchSize = 100;
H = 10; % number of hidden layers
labelHotBit = hotBit(labels, k); %hot bit encoded labels
hiddenWeights = rand(H, d); %first layer weights
hiddenBias = zeros(H, 1);
outputWeights = rand(k, H); %second layer weights
outputBias = zeros(k, 1);
tol = 10e-5;
epochs = 500;
% cost = 2;
% lastCost = 1;
losses = [];
%%
% works but gets stuck on some gradients, so we need to retry until its
% unstuck
for epoch = 1:epochs
    % Shuffle data 
    indices = randperm(n);
    X_shuffled = imgs(indices, :);
    Y_shuffled = labelHotBit(indices, :);
    
    for i = 1:batchSize:n
        % Select the current batch
        % m(i) = floor(rand(1)*n + 1);
        % inputVector = imgs(m(i), :)';
        endIndex = min(i + batchSize - 1, n);
        inputBatch = X_shuffled(i:endIndex, :)'; 
        targetBatch = Y_shuffled(i:endIndex, :)';
        
        % Forward pass for the batch
        % g = imgBatch * w1;
        % y2 = sigmoid(g);
        % u = y2 * w2;
        % z2 = softmax(u);
        hiddenActualInput = hiddenWeights*inputBatch + hiddenBias;
        hiddenOutputVector = sigmoid(hiddenActualInput);
        outputActualInput = outputWeights*hiddenOutputVector + outputBias;
        outputVector = softmax(outputActualInput);
        L = -sum(log(sum(outputVector .* targetBatch, 1))) / batchSize
        losses(end+1) = L;
        % targetVector = t(m(i), :)';


        % Backward pass
        % d_out = (z2 - labelBatch) / batchSize; % Normalize by batch size
        % grad_w2 = y2' * d_out;
        % d_hidden = (d_out * w2') .* d_sigmoid(g);
        % grad_w1 = imgBatch' * d_hidden;
        d_outputVector = (outputVector - targetBatch);
        d_outputWeights = d_outputVector * hiddenOutputVector';
        d_outputBias = sum(d_outputVector, 2) / batchSize;

        d_hiddenActualInput = outputWeights'*d_outputVector;
        d_hiddenOutputVector = d_hiddenActualInput.*d_sigmoid(hiddenOutputVector);
        d_sigmoid(hiddenOutputVector)
        
        %VANISHING GRADIENT, THIS IS ZERO FOR SOME REASONFUSAJCKL
        d_hiddenWeights = d_hiddenOutputVector * inputBatch' / batchSize;
        d_hiddenBias = sum(d_hiddenOutputVector, 2) / batchSize;

        % update weights
        hiddenWeights = hiddenWeights - lr*d_hiddenWeights;
        hiddenBias = hiddenBias - lr*d_hiddenBias;

        outputWeights = outputWeights - lr*d_outputWeights;
        outputBias = outputBias - lr*d_outputBias;
        outputWeights;
      
    end
    % cost = crossEntropyCostDouble(t, forward_pass_all(imgs, hiddenWeights', outputWeights'))
    % percentError = calcErrorDouble(labels, forward_pass_all(imgs, hiddenWeights', outputWeights'));
end

%%



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
    exp_a = exp(a);
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

function y = forward_pass_all(imgs, w1, w2)
    % Forward pass for all samples, used to compute the cost
    g = imgs * w1;
    y2 = sigmoid(g);
    u = y2 * w2;
    y = softmax(u);
end