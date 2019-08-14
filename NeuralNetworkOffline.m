clearvars; close all; clc;
%% Neural Network Implementation for offline
clearvars; close all; clc;
% Parameters 
hLayers = 2;                     % Hidden Layers
InputSize = 1;
OutputSize = 1;
Neurons = [InputSize 100 100 OutputSize];         % Neurons per layers (input, hidden, output)
ActFuncType = 1;             % Activation Function: 1: Sigmoid. 2: Lineal TODO: add more functions
DataSize = 20;
lrate = 0.00001;
NumEpochs = 1000;

if hLayers ~= length(Neurons) - 2
    disp('hLayers is different than neurons defined');
end

% Initialization
% Gaussian weight initialization
W1 = randn(Neurons(1),Neurons(2));%, ones(Neurons(1),1)];
W2 = randn(Neurons(2),Neurons(3));%, ones(Neurons(2),1)];
W3 = randn(Neurons(3),Neurons(4));%, ones(Neurons(3),1)];

CostHistory = zeros(NumEpochs,1);

% Random Dataset Generation
%inputs = [3, 5;5, 1;10, 2;4, 3];%randn(DataSize,InputSize);         % row (input)
%outputs = [75;82; 95; 87]/10;%randn(DataSize, OutputSize);         % row (Output)

inputs = linspace(-10,10,40)';%randn(DataSize,InputSize);         % row (input)
outputs = inputs.^2/4 + sin(inputs);%randn(DataSize, OutputSize);         % row (Output)


for epoch = 1:NumEpochs
    if mod(epoch,100) == 0
        disp(['It is epoch ', num2str(epoch)]);
    end
    % Feedforward Stage
%     z_2 = inputs * W1;
%     a_2 = ActFunc(z_2, ActFuncType);
% 
%     z_3 = a_2 * W2;
%     a_3 = ActFunc(z_3, ActFuncType);
% 
%     z_4 = a_3 * W3;
%     y_hat = ActFunc(z_4, 2);

    [y_hat, z_2, a_2, z_3, a_3, z_4] = Feedforward(inputs, W1, W2, W3, ActFuncType);

    % Cost Function Calculation
    J = CostFunction(outputs, y_hat);
    CostHistory(epoch) = J;

    % Backpropagation Stage
    %   Gradient Calculation
    
%     delta4 =  -(outputs - y_hat).* ActFuncPrime(z_4, 2);
%     dJdW3 = a_3'*delta4;
%     delta3 = (delta4 * W3').* ActFuncPrime(z_3, ActFuncType);
%     dJdW2 = a_2'*delta3;
%     delta2 = (delta3 * W2').* ActFuncPrime(z_2, ActFuncType);
%     dJdW1 = inputs'*delta2;
    [dJdW3, dJdW2, dJdW1] = Backpropagation(inputs, outputs, y_hat, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType);
    %   Updating Weights
    W1 = W1 - lrate * dJdW1;
    W2 = W2 - lrate * dJdW2;
    W3 = W3 - lrate * dJdW3;
end

f = figure(1);
plot(CostHistory(10:end));
    
    z_2 = linspace(-6,6,40)' * W1;
    a_2 = ActFunc(z_2, ActFuncType);

    z_3 = a_2 * W2;
    a_3 = ActFunc(z_3, ActFuncType);

    z_4 = a_3 * W3;
    y = ActFunc(z_4, 2);
    
function [y_hat, z_2, a_2, z_3, a_3, z_4] = Feedforward(inputs, W1, W2, W3, ActFuncType)
    ActFunc = @(z, ActFuncType) (ActFuncType == 1) * 1./(1+exp(-z)) + (ActFuncType == 2) * z;

    z_2 = inputs * W1;
    a_2 = ActFunc(z_2, ActFuncType);

    z_3 = a_2 * W2;
    a_3 = ActFunc(z_3, ActFuncType);

    z_4 = a_3 * W3;
    y_hat = ActFunc(z_4, 2);
end

function [dJdW3, dJdW2, dJdW1] = Backpropagation(inputs, outputs, y_hat, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType)
    ActFunc = @(z, ActFuncType) (ActFuncType == 1) * 1./(1+exp(-z)) + (ActFuncType == 2) * z;
    ActFuncPrime = @(z, ActFuncType) (ActFuncType == 1) * ActFunc(z, ActFuncType) .* (1 - ActFunc(z, ActFuncType)) + (ActFuncType == 2) * ones(size(z));
    delta4 =  -(outputs - y_hat).* ActFuncPrime(z_4, 2);
    dJdW3 = a_3'*delta4;
    delta3 = (delta4 * W3').* ActFuncPrime(z_3, ActFuncType);
    dJdW2 = a_2'*delta3;
    delta2 = (delta3 * W2').* ActFuncPrime(z_2, ActFuncType);
    dJdW1 = inputs'*delta2;
end


function J = CostFunction(output, y_hat)
    % Function performs MSE
    J = 0.5*(output - y_hat)'*(output - y_hat);
end

% function a_prime = ActFuncPrime(z, ActFuncType)
%     a_prime = 0;
%     if ActFuncType == 1
%         a_prime = ActFunc(z, ActFuncType) .* (1 - ActFunc(z, ActFuncType));
%     end
%     if ActFuncType == 2
%         a_prime = ones(size(z));
%     end
% end