clearvars; close all; clc;
%% Neural Network Implementation for offline
clearvars; close all; clc;
% Parameters 
hLayers = 2;                     % Hidden Layers
InputSize = 1;
OutputSize = 4;
Neurons = [InputSize 100 100 OutputSize];         % Neurons per layers (input, hidden, output)
ActFuncType = 1;             % Activation Function: 1: Sigmoid. 2: Lineal TODO: add more functions
DataSize = 20;
lrate = 0.0001;
NumEpochs = 10000;

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
%
%inputs = linspace(-10,10,40)';%randn(DataSize,InputSize);         % row (input)
%outputs = inputs.^2/4 + sin(inputs);%randn(DataSize, OutputSize);         % row (Output)

% Data
theta = csvread('PendulumData/SwingUpData1/theta'); theta = theta(:,2);
theta_hat = csvread('PendulumData/SwingUpData1_sim/theta'); theta_hat = theta_hat';

alpha = csvread('PendulumData/SwingUpData1/alpha'); alpha = alpha(:,2);
alpha_hat = csvread('PendulumData/SwingUpData1_sim/alpha'); alpha_hat = alpha_hat';

thetadot = csvread('PendulumData/SwingUpData1/thetadot'); thetadot = thetadot(:,2);
thetadot_hat = csvread('PendulumData/SwingUpData1_sim/thetadot'); thetadot_hat = thetadot_hat';

alphadot = csvread('PendulumData/SwingUpData1/alphadot'); alphadot = alphadot(:,2);
alphadot_hat = csvread('PendulumData/SwingUpData1_sim/alphadot'); alphadot_hat = alphadot_hat';

Vin = csvread('PendulumData/SwingUpData1/Vin'); Vin = Vin(:,2);
%Vin_hat = csvread('PendulumData/SwingUpData1_sim/Vin'); Vin_hat = Vin_hat';

inputs = Vin;
inputs_hat = Vin;
outputs = [theta, alpha,thetadot,alphadot];
outputs_hat = [theta_hat, alpha_hat,thetadot_hat,alphadot_hat];

%%
for epoch = 1:NumEpochs
    if mod(epoch,100) == 0
        disp(['It is epoch ', num2str(epoch)]);
    end
    % Feedforward Stage
    z_2 = inputs * W1;
    a_2 = ActFunc(z_2, ActFuncType);

    z_3 = a_2 * W2;
    a_3 = ActFunc(z_3, ActFuncType);

    z_4 = a_3 * W3;
    y_hat = ActFunc(z_4, 2);

    % Cost Function Calculation
    J = CostFunction(outputs, y_hat);
    CostHistory(epoch) = J;

    % Backpropagation Stage
    %   Gradient Calculation
    delta4 =  -(outputs - y_hat).* ActFuncPrime(z_4, 2);
    dJdW3 = a_3'*delta4;
    delta3 = (delta4 * W3').* ActFuncPrime(z_3, ActFuncType);
    dJdW2 = a_2'*delta3;
    delta2 = (delta3 * W2').* ActFuncPrime(z_2, ActFuncType);
    dJdW1 = inputs'*delta2;

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

function a = ActFunc(z, ActFuncType)
    a = 0;
    if ActFuncType == 1
        a = 1./(1+exp(-z));
    end
    if ActFuncType == 2
       a = z; 
    end
end

function J = CostFunction(output, y_hat)
    % Function performs MSE
    J = 0.5*(output - y_hat)'*(output - y_hat);
end

function a_prime = ActFuncPrime(z, ActFuncType)
    a_prime = 0;
    if ActFuncType == 1
        a_prime = ActFunc(z, ActFuncType) .* (1 - ActFunc(z, ActFuncType));
    end
    if ActFuncType == 2
        a_prime = ones(size(z));
    end
end