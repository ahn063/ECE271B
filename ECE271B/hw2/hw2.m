% Andy Nguyen
% 2/13/2024
% ECE 271B

%% Setup
close all; clc;
if ~(exist('imgs', 'var') && exist('labels', 'var') && exist('imgs_test', 'var') && exist('labels_test', 'var'))  
    [imgs, labels] = readMNIST('training set\train-images-idx3-ubyte', 'training set\train-labels-idx1-ubyte', 60000, 0);
    [imgs_test, labels_test] = readMNIST('test set\t10k-images-idx3-ubyte', 'test set\t10k-labels-idx1-ubyte', 10000, 0);
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

tol = 1e-5;
lr = 1e-5; %learning rate
totalIter = 1000;

%train
while(abs(costVector(end) - costVector(end-1)) > tol)
    [w, cost, y] = one_layer(imgs, labels, w, k, lr);
    iterNum = iterNum+1;
    costVector(iterNum) = cost;
    training_error(iterNum) = calcError(imgs, labels, w);
    test_error(iterNum) = calcError(imgs_test, labels_test, w);
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





function [w, cost, y] = one_layer(imgs, labels, w, numClasses, learningRate)
    a = imgs*w; %should be n x k
    y = softmax(a); % n x k, calculates score of each example
    t = hotBit(labels, numClasses);
    cost = crossEntropyCost(t,y);
    delta_w = backprop(t, y, imgs);
    w = w + learningRate*delta_w; % d x k
end

function output = sigmoid(u)
    output = 1/(1+exp(-u));
end

function output = ReLU(u)
    %u is vector
    output = max(max(0, u));
end

function output = softmax(a)
%     exp_a = exp(a - max(a, [], 2)); % Improve numerical stability
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

function percentError = calcError(imgs, labels, w)
    totalN = size(imgs, 1);
    y = softmax(imgs*w);
    [~, predicted_labels] = max(y, [], 2);
    predicted_labels = predicted_labels - 1;
    percentCorrect = sum(predicted_labels==labels) / totalN;
    percentError = 1-percentCorrect;
end