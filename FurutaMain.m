function FurutaMain
%% Double Inverted Pendulum - Function Approximation - Q-learning
% This algorithm combines ...

clear all; close all; clc;

%% Reaching the upright position stage parameters
lambda = 0.2;                                   % For state initialization
normalAngle = @(angle) abs(2*pi - abs(angle));  % Normalization function 
TLim = 6; % Torque
actions = [-TLim 0 TLim];                       % Only 3 options, Full blast one way, the other way, and off
bestSwingUp = zeros(1000,1);  % Large vector, so it is replaced in the first success
angleRange = 0.1;           % Switch to second controller when q reaches the ball of radius angleRange
rateRange = 0.5;          % Switch to second controller when qd reaches the ball of radius rateRange

% -------- Q-learning -------- %
maxep = 2000;           % Set max number of episodes
tSteps = 800;           % Set max number of iterations because there are undefined states
gamma = 0.99;           % Discount rate
Rfunc = @(q, qdot) -((pi-abs(q)).^1 + 0.2*(abs(qdot).^2));  % Reward function
prob = 0.8;             % Starting probability
epsilon = prob;         % Probability of picking random action vs estimated best action
epsilonDecay = 0.999;   % Decay after each episode
dt = 0.05;              % Timestep of integration. Each substep lasts this long.
success = false;        
bonus = 0;

% -------- NN function approximation --------- % 
hLayers = 2;                    % Hidden Layers. If it varies. More weights must be added.
InputSize = 2;                  % Input size
OutputSize = length(actions);                 % Output size
Neurons = [InputSize 32 128 OutputSize];         % Neurons per layers (input, hidden1, hidden2, output)
ActFuncType = 1;                % Activation Function: 1: Sigmoid. 2: Lineal. TODO: add more functions
lrate = 0.00001;                 % Learning rate for NN

% Gaussian weight initialization
W1 = randn(Neurons(1),Neurons(2));  % Weight for input to layer 1
W2 = randn(Neurons(2),Neurons(3));  % Weight for layer 1 to layer 2
W3 = randn(Neurons(3),Neurons(4));  % Weight for layer 2 to output

CostHistory = zeros(tSteps,maxep);  % To save cost for NN for each episode
RewardHistory = zeros(tSteps,maxep);  % To save reward for trajectory for each episode

FallOverCount = 0;
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
    
    %sarsdone = zeros(7, tSteps)

    NNinput = zeros(tSteps, length(state) );
    NNoutput = zeros(tSteps,length(actions) );

    for iter = 1:tSteps    
        % -------- Feed to obtain for all the actions -------- %
        [q_v , ~, ~, ~, ~, ~] = Feedforward(state, W1, W2, W3, ActFuncType);

        NNoutput(iter,:) = q_v; 	% Save output for Backpropagation
        
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
        
        % Calculate qhat max from the observation
        [q_v , ~, ~, ~, ~, ~] = Feedforward(state, W1, W2, W3, ActFuncType);
        qhat_obs = max(q_v);    
        %--------------------------------------------------------%
        
        % If the limit was reached, episode is defined as succesful
        if ( (obs(1) > pi - angleRange) && (obs(1) < pi + angleRange)  ) && ( abs(obs(2))<rateRange )
            % If we've reached upright with no velocity (within some margin), end this episode
            success = true;
            % Give a bonus for getting there
            bonus = 40; 
            % Save the actions that led the agent to the objective
            % Only overwrite it if in this episode, less actions were
            % needed
            if length(bestSwingUp) > iter
                bestSwingUp = takenA(1:iter);
                bestPosInit = posInit;
            end
            break;
        else
            bonus = 0;
            success = false;
        end
        
        % Save Reward
        r = Rfunc(obs(1),obs(2));
        RewardHistory(iter, episodes) = r;

        % Save input for backpropagation
        NNinput(iter,:) = state;
        if success
            NNoutput(iter, stateA) = r;
        else
            NNoutput(iter, stateA) = r + gamma*qhat_obs + bonus;
        end
        %--------------------------------------------------------%
        
        % Save Cost
        J = CostFunction(NNoutput(iter,stateA), qhat);
        CostHistory(iter, episodes) = J;

        % Update State
        state = obs;

		% Save state        
        simS(1, iter) = state(1);
        simS(2, iter) = state(2);
        
         % epsilon update
        epsilon = epsilon*epsilonDecay;

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
        
        
    end % For loop
   
    
    if success
        disp(['Objective reached on episode No. ', int2str(episodes)]);

    else
        if (mod(episodes,100) == 0)
            disp(['Did not reach the objective on episode No.', int2str(episodes)])
            pause(0.00000001)% Pause needed for printing messages
        end
    end
    
    % Save only before loop breaks
    NNinput = NNinput(1:iter,:);
    NNoutput = NNoutput(1:iter,:);
    
    % Backpropagation
    [NNpred , z_2, a_2, z_3, a_3, z_4]= Feedforward(NNinput, W1, W2, W3, ActFuncType);
    [dJdW3, dJdW2, dJdW1] = Backpropagation(NNinput, NNoutput, NNpred, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType);

    %   Updating Weights
    W1 = W1 - lrate * dJdW1;
    W2 = W2 - lrate * dJdW2;
    W3 = W3 - lrate * dJdW3;
    
    success = false;
    epsilon = prob;
    
end % for Loop

disp(['First stage concluded in ', int2str(episodes), ' episodes']);

%% Save weights and History from prior stage
W1_SwingUp = W1;
W2_SwingUp = W2;
W3_SwingUp = W3;


CostHistory_SwingUp = CostHistory;
RewardHistory_SwingUp = RewardHistory;

%% Take best Actions
bestA = bestSwingUp;
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

%%
% Ploting phase portrait
xd_sp = linspace(-8,8,50);
x_sp = linspace(0,2*pi,50);
[x_grid, xd_grid] = meshgrid(x_sp,xd_sp);
xdnT_grid = zeros(size(x_grid));
xd0T_grid = zeros(size(x_grid));
xdpT_grid = zeros(size(x_grid));
xddnT_grid = zeros(size(x_grid));
xdd0T_grid = zeros(size(x_grid));
xddpT_grid = zeros(size(x_grid));
for i = 1:size(xd_grid,1)*size(xd_grid,2)
    state_temp = [x_grid(i) , xd_grid(i)];
    [q_temp , ~, ~, ~, ~, ~]= Feedforward(state_temp, W1_SwingUp, W2_SwingUp, W3_SwingUp, ActFuncType);
    [~,ActIndex_temp] = max(q_temp);
    stated_temp = Dynamics(state_temp,actions(ActIndex_temp));
    xddnT_grid(i) = stated_temp(2)*(ActIndex_temp==1);
    xdd0T_grid(i) = stated_temp(2)*(ActIndex_temp==2);
    xddpT_grid(i) = stated_temp(2)*(ActIndex_temp==3);
    xdnT_grid(i) = xd_grid(i)*(ActIndex_temp==1);
    xd0T_grid(i) = xd_grid(i)*(ActIndex_temp==2);
    xdpT_grid(i) = xd_grid(i)*(ActIndex_temp==3);
end

%%
PlotScale = 50;
figure;
quiver(x_grid,xd_grid,xdnT_grid/PlotScale,xddnT_grid/PlotScale,'LineWidth',1.1,'AutoScale','off','MarkerSize',20);hold on;
quiver(x_grid,xd_grid,xd0T_grid/PlotScale,xdd0T_grid/PlotScale,'LineWidth',1.1,'AutoScale','off','MarkerSize',20);
quiver(x_grid,xd_grid,xdpT_grid/PlotScale,xddpT_grid/PlotScale,'LineWidth',1.1,'AutoScale','off','MarkerSize',20);
plot(simS(:,1),simS(:,2)); hold off;
legend('- \tau','zero \tau','+ \tau');
xlabel('x'), ylabel('dot x')

%% Close Control parameters

actions = [-0.5 0.5]; 

%-- Q-learning --%
tSteps = 300; 		% Set number of iterations because there are undefined states
gamma = 0.7; 		% Discount rate
ballRadius = pi/2-0.9;
fball = @(x) 5*tan(abs(pi/ballRadius*x-3*pi/2));
fballabs = @(q) (1+sign(fball(q)))/2 ;%+ max(fball(min(q,ballRadius-0.0001)) ,-15)*(1-sign(fball(q)))/2;  min(fball(q),15)*
Rfunc = @(q) fballabs(abs(q-pi));
%Rfunc = @(q) min(1/abs(q-pi),15); %-2*abs(q-pi)^1.3;
epsilonDecay = 0.999;

% General variables
maxep = 500;
cFlag = false;
resetCount = -1;

%% Second Part - Control

InputSize = 2;				                  % Input size
OutputSize = length(actions);                 % Output size

Neurons = [InputSize 50 100 OutputSize];         % Neurons per layers (input, hidden1, hidden2, output)

% Gaussian weight initialization
W1 = randn(Neurons(1),Neurons(2));
W2 = randn(Neurons(2),Neurons(3));
W3 = randn(Neurons(3),Neurons(4));

CostHistory = zeros(tSteps,maxep);  % To save cost for NN for each episode
RewardHistory = zeros(tSteps,maxep);  % To save reward for trajectory for each episode

%%
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
        simS2 = zeros(length(state),tSteps);
        takenStable = zeros(1, tSteps);

        NNinput = zeros(tSteps, length(state) );
        NNoutput = zeros(tSteps,length(actions) );

        for iter = 1:tSteps
        	% -------- Feed to obtain for all the actions -------- %
            [q_v , ~, ~, ~, ~, ~] = Feedforward(state, W1, W2, W3, ActFuncType);

            NNoutput(iter,:) = q_v;  % Save output for Backpropagation
            
            if (rand()>epsilon) % Choose action from epsilon-greedy policy
               [qhat, stateA] = max(q_v);
            else % Random action!
                qhat = max(q_v);
                stateA = randi(length(actions),1); 
            end

            T = actions(stateA);
            takenStable(iter) = actions(stateA);
            
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

            % Calculate qhat max from the observation
            [q_v , ~, ~, ~, ~, ~] = Feedforward(state, W1, W2, W3, ActFuncType);
            qhat_obs = max(q_v);        

            %--------------------------------------------------------%

            % This time, the reward is only given when the agent reaches its
            % upright position with almost no error margin
            %if ( (obs(1) > pi-0.1) && (obs(1) < pi+0.1)  )
            %    bonus = 20;
            %else % and it is penalized, otherwise
            %    bonus = 0;
            %end

            if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  ) || ( abs(state(2)) > 8 )
                break;
            end
            
            % Save Reward

            r = Rfunc(obs(1));
            RewardHistory(iter, episodes) = r;

        	% Save input and output for backpropagation
            NNinput(iter,:) = state;
            NNoutput(iter, stateA) = r + gamma*qhat_obs;% + bonus;
        	
        	% Save Cost
            J = CostFunction(NNoutput(iter,stateA), qhat);
            CostHistory(iter, episodes) = J;
                   
            %--------------------------------------------------------%

            % Update State
            state = obs;

	        % Save state        
            simS2(1, iter) = state(1);
            simS2(2, iter) = state(2);

            %if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  ) || ( abs(state(2)) > 8 )
            %    break;
            %end

        end % For loop

        takenStable = takenStable(1:iter);
        NNinput = NNinput(1:iter,:);
        NNoutput = NNoutput(1:iter,:);
        
        
        [NNpred , z_2, a_2, z_3, a_3, z_4]= Feedforward(NNinput, W1, W2, W3, ActFuncType);
        [dJdW3, dJdW2, dJdW1] = Backpropagation(NNinput, NNoutput, NNpred, z_4, a_3,  z_3, a_2, z_2, W3, W2, ActFuncType);


        %   Updating Weights
        W1 = W1 - lrate * dJdW1;
        W2 = W2 - lrate * dJdW2;
        W3 = W3 - lrate * dJdW3;
    
        stats(episodes) = iter;
        
        % Count times the pendulum was controlled successfully
        if (stats(episodes) == tSteps)
            breakCount = breakCount + 1;
        else
            breakCount = 0;
        end
        
        disp(['Episode No.', int2str(episodes)])
        epsilon = epsilon*epsilonDecay; % Reduce exploration probability
        episodes = episodes + 1;

        if breakCount >= 10
            cFlag = true;
%           break;
        end

    end % While Loop    
    disp(['Max iter ', num2str(max(stats)),' Last Iter ', num2str(stats(end))]);
    
    cFlag = true;
end % While Loop



%% Completing best actions
bestA = [bestSwingUp takenStable];
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
% Plot

figure;
plot(1:maxep, stats, 'b');
xlabel('Episodes');
ylabel('Iterations');
title('Second Stage episodes vs. iterations');
axis([0 1000 0 350]);

figure;
plot((0:dt:(time-1)*dt), simS(:,1), (0:dt:(time-1)*dt), simS(:,2));
xlabel('time (s)');
ylabel('Angle (rad) / Rate (rad/s)');
title('Angle/Rate vs. time');
hl = legend('Angle ($\theta$)', 'Rate ($\dot{\theta}$)');
set(hl, 'Interpreter', 'latex');
axis([0 (time-1)*dt -2*pi-0.5 2*pi+0.5]);

%%
% Simulate
parameters = [300 100 750 500];


simulate1Pend(parameters, simS);



end

% Reinforcement learning function

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
