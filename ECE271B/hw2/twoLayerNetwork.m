function twoLayerNetwork(H, eta, X_train, Y_train, numIterations)
    % Convert Y_train to one-hot encoding
    Y_train_onehot = hotBit(Y_train, 10)% Assuming Y_train labels are 0-based
    
    % Initialize network parameters
    inputSize = size(X_train, 2); % 784 for MNIST
    numClasses = size(Y_train_onehot, 1); % 10 for MNIST
    W1 = randn(H, inputSize) * 0.01;
    b1 = zeros(H, 1);
    W2 = randn(numClasses, H) * 0.01;
    b2 = zeros(numClasses, 1);
    
    % Preallocate error storage
    errors = zeros(1, numIterations);
    
    for t = 1:numIterations
        % Forward pass
        Z1 = W1 * X_train' + b1;
        A1 = 1 ./ (1 + exp(-Z1)); % Sigmoid activation
        Z2 = W2 * A1 + b2;
        A2 = exp(Z2) ./ sum(exp(Z2), 1); % Softmax activation
        
        % Compute cross-entropy loss
        L = -sum(log(sum(A2 .* Y_train_onehot, 1))) / size(X_train, 1);
        
        % Backward pass
        dZ2 = A2 - Y_train_onehot;
        dW2 = dZ2 * A1' / size(X_train, 1);
        db2 = sum(dZ2, 2) / size(X_train, 1);
        dA1 = W2' * dZ2;
        dZ1 = dA1 .* A1 .* (1 - A1);
        dW1 = dZ1 * X_train / size(X_train, 1);
        db1 = sum(dZ1, 2) / size(X_train, 1);
        
        % Update parameters
        W1 = W1 - eta * dW1;
        b1 = b1 - eta * db1;
        W2 = W2 - eta * dW2;
        b2 = b2 - eta * db2;
        
        % Store error
        errors(t) = L;
    end
    
    % Plot training error
    figure;
    plot(1:numIterations, errors);
    title('Training Error over Iterations');
    xlabel('Iteration');
    ylabel('Cross-Entropy Loss');
end

function hotBitVector = hotBit(v, numClasses)
    %v is n x 1, where n = numExamples
    n = size(v,1);
    hotBitVector = zeros(n, numClasses);
    for i = 1:n
        hotBitVector(i, v(i)+1) = 1;
    end
end
