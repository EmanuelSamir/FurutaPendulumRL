function FurutaMain
%% Double Inverted Pendulum - Function Approximation - Q-learning
% This algorithm combines ...

clear all; close all; clc;

%% Reaching the upright position stage parameters
lambda = 0.2;                                   % For state initialization
normalAngle = @(angle) abs(2*pi - abs(angle));  % Normalization function 
TLim = 9; % Torque
actions = [-TLim 0 TLim];                       % Only 3 optionsFull blast one way, the other way, and off
bestA = zeros(1000,1);  % Large vector, so it is replaced in the first success
angleRange = 0.07;           % Switch to second controller when q reaches the ball of radius angleRange
rateRange = 0.1;          % Switch to second controller when qd reaches the ball of radius rateRange

% -------- Function Approximation -------- %
% alpha = 0.000001;                             % Step size for function approximation
% w = zeros(length(getX1([0 0],1)),1); % Weight parameters for features
% Rfunc = @(x,xdot)(-(abs(x)).^2 - (abs(xdot)).^2); % Reward function

% -------- Q-learning -------- %
maxep = 100;           % Set max number of episodes
tSteps = 800;           % Set max number of iterations because there are undefined states
gamma = 0.99;           % Discount rate
Rfunc = @(q, qdot) -((pi-abs(q)).^1 + 0.2*(abs(qdot).^3.5));  % Reward function
prob = 0.8;             % Starting probability
epsilon = prob;         % Probability of picking random action vs estimated best action
epsilonDecay = 0.999;   % Decay after each episode
dt = 0.05;              % Timestep of integration. Each substep lasts this long.
success = false;        
bonus = 0;

% -------- NN function approximation --------- % 
hLayers = 2;                    % Hidden Layers. If it varies. More weights must be added.
InputSize = 3;                  % Input size
OutputSize = 1;                 % Output size
Neurons = [InputSize 40 40 OutputSize];         % Neurons per layers (input, hidden1, hidden2, output)
ActFuncType = 1;                % Activation Function: 1: Sigmoid. 2: Lineal. TODO: add more functions
lrate = 0.00001;                 % Learning rate for NN

% Gaussian weight initialization
W1 = randn(Neurons(1),Neurons(2));  % Weight for input to layer 1
W2 = randn(Neurons(2),Neurons(3));  % Weight for layer 1 to layer 2
W3 = randn(Neurons(3),Neurons(4));  % Weight for layer 2 to output

CostHistory = zeros(tSteps,maxep);  % To save cost for NN for each episode
RewardHistory = zeros(tSteps,maxep);  % To save reward for trajectory for each episode

FallOverCount = 1;
HighRateCount = 0;

%% First Part - Reach Upright Position
for episodes = 1:maxep
    % Reset initial state
    
    % [angle1 rate1]
    state = [ unifrnd(-lambda,lambda) 0 ];
    posInit = state;
    
    % Reset simulation matrices
    simS = zeros(length(state),tSteps);
    takenA = zeros(1, tSteps);
    
    for iter = 1:tSteps    
        % -------- Pick an action -------- %
        q_v = zeros(1,length(actions));
        for i = 1:length(actions)
            NNinputs_temp = [state, actions(i)];
            [q_temp , ~, ~, ~, ~, ~]= Feedforward(NNinputs_temp, W1, W2, W3, ActFuncType);
            q_v(i) = q_temp;
        end

        % Choose action from epsilon-greedy policy
        if (rand()>epsilon)     
           [qhat, stateA] = max(q_v);
        else % Random action!
            qhat = max(q_v);
            stateA = randi(length(actions),1); 
        end
            
        T = actions(stateA);
        takenA(iter) = actions(stateA);
        %--------------------------------------------------------%
        
        %-- Observation --%
        obs = state;
        % Numerical integration
        for i = 1:2
            k1 = Dynamics(obs,T);
            k2 = Dynamics(obs+dt/2*k1,T);
            k3 = Dynamics(obs+dt/2*k2,T);
            k4 = Dynamics(obs+dt*k3,T);
          
            obs = obs + dt/6*(k1 + 2*k2 + 2*k3 + k4);
            % All states wrapped to 2pi
            if (obs(1) < 0) || (obs(1) > 2*pi)
                obs(1) = normalAngle(obs(1));
            end
            
        end
        %--------------------------------------------------------%
        
        % Calculate qhat from the observation
        q_v = zeros(1,length(actions));
        for i = 1:length(actions)
            NNinputs_temp = [obs, actions(i)];
            [q_temp , ~, ~, ~, ~, ~] = Feedforward(NNinputs_temp, W1, W2, W3, ActFuncType);
            q_v(i) = q_temp;
        end        
        qhat_obs = max(q_v);    
        %--------------------------------------------------------%
        
        % If the limit was reached, episode is defined as succesful
        if ( (obs(1) > pi - angleRange) && (obs(1) < pi + angleRange)  ) && ( abs(obs(2))<rateRange )
            % If we've reached upright with no velocity (within some margin), end this episode
            success = true;
            % Give a bonus for getting there
            bonus = 200; 
            % Save the actions that led the agent to the objective
            % Only overwrite it if in this episode, less actions were
            % needed
            if length(bestA) > iter %length(takenA)
                bestA = takenA(1:iter);
                bestPosInit = posInit;
            end
            disp(length(bestA));
            break
        else
            bonus = 0;
        end
        
        % -------- Update weight parameters -------- %
        % Recover best pair state action chosen
        StaAct_k = [state,actions(stateA)];
        % Recover insight parameters for the best pair
        [~, z_2, a_2, z_3, a_3, z_4] = Feedforward(StaAct_k, W1, W2, W3, ActFuncType);  
        % Save Reward
        r = Rfunc(obs(1),obs(2));
        RewardHistory(iter, episodes) = r;

        % Using the observation and reward as an improved q
        qest = r + gamma*qhat_obs + bonus;

        % Using Backpropagation
        [dJdW3, dJdW2, dJdW1] = Backpropagation(StaAct_k, qest, qhat, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType);

        %   Updating Weights
        W1 = W1 - lrate * dJdW1;
        W2 = W2 - lrate * dJdW2;
        W3 = W3 - lrate * dJdW3;
        %--------------------------------------------------------%
        % Save Cost
        J = CostFunction(qest, qhat);
        CostHistory(iter, episodes) = J;

        % Update State
        state = obs;
        % simS = [simS; state(1:2)];
        simS(1, iter) = state(1);
        simS(2, iter) = state(2);

        % If the objective was reached
        if success
            % But the pendulum falls over
            if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  )
                FallOverCount = FallOverCount + 1;
                break; % Stop
            end
        end
        
        % Stop if the pendulum starts spinning at a high rate
        if ( abs(state(2)) > 8 )
            HighRateCount = HighRateCount +1;
            break;
        end
        
        epsilon = epsilon*epsilonDecay;
        
    end % For loop
    
    if success
        disp(['Objective reached on episode No. ', int2str(episodes)]);
        %m=input('Do you want to continue, Y/N [Y]:','s');
        %if m=='N'
        %    break
        %end
    else
        if (mod(episodes,100) == 0)
            disp(['Did not reach the objective on episode No.', int2str(episodes)])
            pause(0.00000001)% Pause needed for printing messages
        end
    end
    
    success = false;
    epsilon = prob;
    
end % for Loop

% -- Add "Convergence" paramter
% If the number of best actions wont go any lower after "x" episodes
% Or
% If the number of best actions is around the same after "x" episodes
% Or
% Number of actions threshold --%

disp(['First stage concluded in ', int2str(episodes), ' episodes']);

%% Take best Actions
lastState = bestPosInit;
simS = [];

for num = 1:length(bestA)
    T = bestA(num);
    
    for i = 1:2
        k1 = Dynamics(lastState,T);
        k2 = Dynamics(lastState+dt/2*k1,T);
        k3 = Dynamics(lastState+dt/2*k2,T);
        k4 = Dynamics(lastState+dt*k3,T);

        lastState = lastState + dt/6*(k1 + 2*k2 + 2*k3 + k4);
        % All states wrapped to 2pi
        if (lastState(1) < 0) || (lastState(1) > 2*pi)
            lastState(1) = normalAngle(lastState(1));
        end
    end
    
    simS = [simS; lastState(1:2)];
    
end % Take Best actions

%% Control parameters

%-- Function Approximation --%
actions = [-0.5 0.5]; % Only 3 options, Full blast one way, the other way, and off

%-- Q-learning --%
tSteps = 300; % Set number of iterations because there are undefined states
gamma = 0.7; % Discount rate
Rfunc = @(q) -(abs(q-pi));
epsilonDecay = 0.9999;

% General variables
maxep = 5000;
cFlag = false;
resetCount = -1;

%% Second Part - Control

% Save weights and History from prior stage
W1_first = W1;
W2_first = W2;
W3_first = W3;

CostHistory_first = CostHistory;
RewardHistory_first = RewardHistory;
%%
Neurons = [InputSize 4 30 OutputSize];         % Neurons per layers (input, hidden1, hidden2, output)

% Gaussian weight initialization
W1 = randn(Neurons(1),Neurons(2));%, ones(Neurons(1),1)];
W2 = randn(Neurons(2),Neurons(3));%, ones(Neurons(2),1)];
W3 = randn(Neurons(3),Neurons(4));%, ones(Neurons(3),1)];


CostHistory = zeros(tSteps,maxep);  % To save cost for NN for each episode
RewardHistory = zeros(tSteps,maxep);  % To save reward for trajectory for each episode


%%
k = 0;
cFlag = false;
while(~cFlag)
    % Restart algorithm, since it did not succesfully controlled the agent
    % in 300 continous episodes
    episodes = 1;
    breakCount = 0;
    stats = zeros(1,maxep);
    resetCount = resetCount + 1;
    epsilon = 0.8;
    
    while(episodes<maxep+1)
        % Reset initial state
        % [angle rate]
        state = lastState;

        % Reset simulation matrices
        % simS2 = [];
        % takenA = [];

        simS2 = zeros(length(state),tSteps);
        takenA = zeros(1, tSteps);


        for iter = 1:tSteps
            %-- Pick an action --%
            q_v = zeros(1,length(actions));
            for i = 1:length(actions)
                NNinputs_temp = [state, actions(i)];
                [q_temp , ~, ~, ~, ~, ~]= Feedforward(NNinputs_temp, W1, W2, W3, ActFuncType);
                q_v(i) = q_temp;
            end

            if (rand()>epsilon) % Choose action from epsilon-greedy policy
               [qhat, stateA] = max(q_temp);
            else % Random action!
                qhat = max(q_temp);
                stateA = randi(length(actions),1); 
            end

            T = actions(stateA);
            
            %--------------------------------------------------------%

            %-- Observation --%
            obs = state;
            % Numerical integration
            for i = 1:2
                k1 = Dynamics(obs,T);
                k2 = Dynamics(obs+dt/2*k1,T);
                k3 = Dynamics(obs+dt/2*k2,T);
                k4 = Dynamics(obs+dt*k3,T);

                obs = obs + dt/6*(k1 + 2*k2 + 2*k3 + k4);
                % All states wrapped to 2pi
                if (obs(1) < 0) || (obs(1) > 2*pi)
                    obs(1) = normalAngle(obs(1));
                end

            end
            %--------------------------------------------------------%

            % Calculate qhat from the observation
            q_v = zeros(1,length(actions));
            for i = 1:length(actions)
                NNinputs_temp = [obs, actions(i)];
                [q_temp , ~, ~, ~, ~, ~] = Feedforward(NNinputs_temp, W1, W2, W3, ActFuncType);
                q_v(i) = q_temp;
            end
            qhat_obs = max(q_temp);        
            %--------------------------------------------------------%

            % This time, the reward is only given when the agent reaches its
            % uprigth position with almost no error margin
            if ( (obs(1) > pi-0.1) && (obs(1) < pi+0.1)  )
                bonus = 20;
            else % and it is penalized, otherwise
                bonus = 0;
            end

            if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  ) || ( abs(state(2)) > 8 )
                bonus = -30;
            end
            %-- Update weight parameters --%
            StaAct_k = [state,actions(stateA)];

            [~, z_2, a_2, z_3, a_3, z_4] = Feedforward(StaAct_k, W1, W2, W3, ActFuncType);

            r = Rfunc(obs(1)) - 0.1*obs(2)^4;
            RewardHistory(iter, episodes) = r;
            qest = r + gamma*qhat_obs + bonus;

            J = CostFunction(qest, qhat);
            CostHistory(iter, episodes) = J;
            
            [dJdW3, dJdW2, dJdW1] = Backpropagation(StaAct_k, qest, qhat, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType);

            %   Updating Weights
            W1 = W1 - lrate * dJdW1;
            W2 = W2 - lrate * dJdW2;
            W3 = W3 - lrate * dJdW3;            
            %--------------------------------------------------------%

            % Update State
            state = obs;

            % simS2 = [simS2; state(1:2)];
            simS2(1, iter) = state(1);
            simS2(2, iter) = state(2);
            takenA(iter) = actions(stateA);
            % takenA = [takenA actions(stateA)];

            if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  ) || ( abs(state(2)) > 8 )
                break;
            end

        end % For loop

        stats(episodes) = iter;
        takenA = takenA(1:iter);
        
        % Count times the pendulum was controlled successfully
        if (stats(episodes) == tSteps)
            breakCount = breakCount + 1;
        else
            breakCount = 0;
        end
        
        %disp(['Episode No.', int2str(episodes)])
        epsilon = epsilon*epsilonDecay; % Reduce exploration probability
        episodes = episodes + 1;

        if breakCount >= 10
            cFlag = true;
%           break;
        end

    end % While Loop    
    %cFlag = true;
    disp(['Max iter ', num2str(max(stats)),' Last Iter ', num2str(stats(end))]);
    
end % While Loop


%% Take complete set of Best Actions
bestA = [bestA takenA];
finalState = bestPosInit;
simS = [];
time = 0;
for num = 1:length(bestA)
    T = bestA(num);
    
    for i = 1:2
        k1 = Dynamics(finalState,T);
        k2 = Dynamics(finalState+dt/2*k1,T);
        k3 = Dynamics(finalState+dt/2*k2,T);
        k4 = Dynamics(finalState+dt*k3,T);

        finalState = finalState + dt/6*(k1 + 2*k2 + 2*k3 + k4);
        % All states wrapped to 2pi
        if (finalState(1) < 0) || (finalState(1) > 2*pi)
            finalState(1) = normalAngle(finalState(1));
        end
        
    end
    
    time = time + 1;
    simS = [simS; finalState(1:2)];
    
end % Take Best actions

%% Plotting and Simulation
% simS = [simS; simS2];
% close all;
% Plot

figure;
plot(1:maxep, stats, 'b');
xlabel('Episodes');
ylabel('Iterations');
title('Second Stage episodes vs. iterations');
axis([0 1000 0 350]);

figure;
plot((1:time)/10, simS(:,1), (1:time)/10, simS(:,2));
xlabel('time (s)');
ylabel('Angle (rad) / Rate (rad/s)');
title('Angle/Rate vs. time');
hl = legend('Angle ($\theta$)', 'Rate ($\dot{\theta}$)');
set(hl, 'Interpreter', 'latex');
axis([0 time/10 -2*pi-0.5 2*pi+0.5]);
%%
% Simulate
parameters = [300 100 750 500];
simulate1Pend(parameters, simS);

end
% Reinforcement learning function

function x = getX1(state, action)
% Features for linear Q-learning Part 1
    x = [ 1 1 ...
        state(1)^2 state(1) ...
        state(2)^2 state(2)... 
        (action*state(1))^2 action*state(1)...
        (action*state(2))^2 action*state(2)];
end

function x = getX2(state, action)
% Features for linear Q-learning Part 2
    x = [ 1 state(1)^2 state(1) ...
        state(2)^2 state(2) ...
        action*(pi-state(1)) ...
        action*state(2)];
        
end

 
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
