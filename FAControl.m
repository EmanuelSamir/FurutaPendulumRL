function FAControl
%% Double Inverted Pendulum - Function Approximation - Q-learning
% This function trains the pendulum for mantaining its vertical position, 
% using another Function Approximation algorithm. It assumes that the agent
% has already reached its upright position, with a error margin of +/- 0.01
% rads for the angle and +/- 0.05 rads/s for the rate.

% Reference:
%
%   FunctionApproximation1Pend

clear all; close all; clc;

%% Determine initial parameters

%-- Function Approximation --%
% Function to normalize angle between 0 and 2pi
normalAngle = @(angle) abs(2*pi - abs(angle));
w = zeros(length(getX([0 0],1)),1); % Weight parameters for features
alpha = 0.0001; % Step size for function approximation
actions = [-0.5 0.5]; % Only 3 options, Full blast one way, the other way, and off

%-- Q-learning --%
tSteps = 300; % Set number of iterations because there are undefined states
gamma = 0.70; % Discount rate
% Rfunc = @(x,xdot)(-(abs(x)).^2 - (abs(xdot)).^2); % Reward function
% Rfunc = @(q, qdot) -((pi-abs(q)).^2 + 0.2*(abs(qdot).^2));
Rfunc = @(q) -(10*norm(q,pi));
prob = 0.45;
epsilon = prob; % Probability of picking random action vs estimated best action
epsilonDecay = 0.99;
dt = 0.05; % Timestep of integration. Each substep lasts this long
success = false;

%% while loop
episodes = 1;
maxep = 5000;
stats = zeros(1,maxep);
breakCount = 0;

while(episodes<maxep+1)
    % Reset initial state
    % [angle rate]
    state = [ pi+unifrnd(-0.01,0.01) 0];
    % Reset simulation matrices
    simS = [];
    
    for iter = 1:tSteps
        
        %-- Pick an action --%
        q_temp = [];
        for i = 1:length(actions)
            q_temp = [q_temp getX(state,actions(i))*w];
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
        q_temp = [];
        for i = 1:length(actions)
            q_temp = [q_temp getX(obs, actions(i))*w];
        end
        qhat_obs = max(q_temp);        
        %--------------------------------------------------------%
        
        % This time, the reward is only given when the agent reaches its
        % uprigth position with almost no error margin
        if ( (obs(1) > pi-0.05) && (obs(1) < pi+0.05)  )
            success = true;
            bonus = 100;
        else % and it is penalized, otherwise
            bonus = 0;
        end
        
        %-- Update weight parameters --%
        % deltaW = alpha*( r' + gamma*max(qhat(s',a',w)) - qhat(s,a,w) )*x(s,a)
        % ' means from the observed state
                      
        deltaW = alpha*( Rfunc(obs(1)) + gamma*qhat_obs - qhat + bonus )*getX(state,actions(stateA));
        w = w + deltaW';
        %--------------------------------------------------------%
        
        % Update State
        %delta = max(delta, abs(qhat_obs - qhat));
        state = obs;
        simS = [simS; state(1:2)];
        
        if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  ) || ( abs(state(2)) > 8 )
            break;
        end
                
    end % For loop
    
    stats(episodes) = iter;
    
    if (stats(episodes) == tSteps)
        breakCount = breakCount + 1;
    else
        breakCount = 0;
    end
    
    if success
        disp(['Objective reached on episode nº ', int2str(episodes)])
        disp('Continue...')

    else
        disp(['Didn´t reach the objective on episode nº ', int2str(episodes)])
        pause(0.00000001)% Pause needed for printing messages
    end
    
    success = false;
    % Reduce exploration probability
    epsilon = epsilon*epsilonDecay;
    episodes = episodes + 1;
    
    if breakCount == 300
%         break
    end
    
end % While Loop

figure;
plot(1:maxep, stats);
disp(['Converged with ', int2str(episodes), ' episodes']);
%% Simulation
% close all;
parameters = [300 100 750 500];
% simS = [state(1) state(2)];
% simS = simS(ceil(end/2):end,1:2);
simulate1Pend(parameters, simS);

end % Reinforcement learning function

function x = getX(state, action)
% Features for linear Q-learning
%         x = [ 1 1 ...
%             state(1)^2 state(1) ...
%             state(2)^2 state(2)... 
%             (action*state(1))^2 action*state(1)...
%             (action*state(2))^2 action*state(2)];
          x = [ 1 state(1)^2 state(1) ...
                state(2)^2 state(2) ...
                action*(pi-state(1)) ...
                action*state(2)];
%          x = [state(1) state(1)^2 state(2) state(2)^2 ...
%              state(1)*state(2) state(1)^2*state(2) ...
%              state(1)*state(2)^2 state(1)^2*state(2)^2 ...
%              state(1)*action state(1)*action^2 state(1)^2*action ...
%              state(2)*action state(2)*action^2 state(2)^2*action];
        
end