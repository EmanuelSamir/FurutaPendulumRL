function FunctionApproximation1Pend
%% Double Inverted Pendulum - Function Approximation - Q-learning
% This algorithm performs infinite episodes with a set number of iterations
% per episode in order to lift up the pendulum to its upright position.

% Reference:
%
%   1. Control of Inverted Double Pendulum using Reinforcement Learning
%       - Fredrik Gustafsson, 2016

clear all; close all; clc;

%% Determine initial parameters

%-- Function Approximation --%
lambda = 0.2;
normalAngle = @(angle) abs(2*pi - abs(angle));
w = zeros(length(getX([0 0],1)),1); % Weight parameters for features
alpha = 0.000001; % Step size for function approximation
TLim = 9; % Torque
actions = [-TLim 0 TLim]; % Only 3 options, Full blast one way, the other way, and off
bestA = zeros(1000,1); % Large vector, so it is replaced in the first success

%-- Q-learning --%
maxep = 2000;
tSteps = 800; % Set number of iterations because there are undefined states
gamma = 0.99; % Discount rate
% Rfunc = @(x,xdot)(-(abs(x)).^2 - (abs(xdot)).^2); % Reward function
Rfunc = @(q, qdot) -((pi-abs(q)).^2 + 0.2*(abs(qdot).^2));
prob = 0.45;
epsilon = prob; % Probability of picking random action vs estimated best action
epsilonDecay = 0.999;
dt = 0.05; % Timestep of integration. Each substep lasts this long
success = false;
bonus = 0;
episodes = 1;

%% while loop

for episodes = 1:maxep
    % Reset initial state
    % [angle1 rate1]
    state = [ unifrnd(-lambda,lambda) 0 ];
    posInit = state;
    % Reset simulation matrices
    simS = [];
    takenA = [];
    
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
        % Numerical integration
        for i = 1:2
            k1 = Dynamics(state,T);
            k2 = Dynamics(state+dt/2*k1,T);
            k3 = Dynamics(state+dt/2*k2,T);
            k4 = Dynamics(state+dt*k3,T);
            
            obs = state + dt/6*(k1 + 2*k2 + 2*k3 + k4);
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
        
        % If the limit was reached epidose is defined as succesful
        if ( (obs(1) > pi-0.01) && (obs(1) < pi+0.01)  ) && ( abs(obs(2))<0.05 )% If we've reached upright with no velocity (within some margin), end this episode
            success = true;
            bonus = 100; % Give a bonus for getting there
            
            % Save the actions that led the agent to the objective
            % Only overwrite it if in this episode, lees actions were
            % needed
            if length(bestA) > length(takenA)
                bestA = takenA;
                bestPosInit = posInit;
            end
            
        else
            bonus = 0;
        end
        
        %-- Update weight parameters --%
        % deltaW = alpha*( r' + gamma*max(qhat(s',a',w)) - qhat(s,a,w) )*x(s,a)
        % ' means from the observed state
                      
        deltaW = alpha*( Rfunc(obs(1),obs(2)) + gamma*qhat_obs - qhat + bonus )*getX(state,actions(stateA));
        w = w + deltaW';
        %--------------------------------------------------------%
        
        % Update State
        %delta = max(delta, abs(qhat_obs - qhat));
        state = obs;
        simS = [simS; state(1:2)];
        %woo = [(state(1) > pi-0.05) && (state(1) < pi+0.05) ( abs(state(2))<0.15 )]
        
        if success
            if ( (state(1) > 3*pi/2) || (state(1) < pi/2)  )
                break;
            end
        end
        if ( abs(state(2)) > 8 )
            break;
        end
        
        epsilon = epsilon*epsilonDecay;
        
        takenA = [takenA actions(stateA)];
        
    end % For loop
    
    if success
        disp(['Objective reached on episode nº ', int2str(episodes)])
        disp('Continue...')     

    else
        disp(['Didn´t reach the objective on episode nº ', int2str(episodes)])
        pause(0.00000001)% Pause needed for printing messages
    end
    
    success = false;
    epsilon = prob;
    
end % for Loop

% -- Add "Convergence" paramter
% If the number of best actions won go any lower after "x" episodes
% Or
% If the number of best actions is around the same after "x" episodes
% Or
% Number of actions threshold

disp(['Converged with ', int2str(episodes), ' episodes']);
%% Take best Actions
simS = [];
state = bestPosInit;

for num = 1:length(bestA)
    T = bestA(num);
    
    for i = 1:2
        k1 = Dynamics(state,T);
        k2 = Dynamics(state+dt/2*k1,T);
        k3 = Dynamics(state+dt/2*k2,T);
        k4 = Dynamics(state+dt*k3,T);

        obs = state + dt/6*(k1 + 2*k2 + 2*k3 + k4);
        % All states wrapped to 2pi
        if (obs(1) < 0) || (obs(1) > 2*pi)
            obs(1) = normalAngle(obs(1));
        end
    end
    
    state = obs;
    simS = [simS; state(1:2)];
end

%% Simulation
close all;
parameters = [300 100 750 500];
% simS = [state(1) state(2)];
simS = simS(ceil(end/2):end,1:2);
simulate1Pend(parameters, simS);

end % Reinforcement learning function

function x = getX(state, action)
% Features for linear Q-learning
    x = [ 1 1 ...
        state(1)^2 state(1) ...
        state(2)^2 state(2)... 
        (action*state(1))^2 action*state(1)...
        (action*state(2))^2 action*state(2)];
%     x = [ 1 ...
%         state(1)^2 state(1) ...
%         state(2)^2 state(2)... 
%         (action/10*state(1))^2 action/10*state(1)...
%         (action/10*state(2))^2 action/10*state(2)];
%     x = [ 1 state(1)^2 state(1) ...
%         state(2)^2 state(2) ...
%         action*(pi-state(1)) ...
%         action*state(2)];
         
end

% Nuevo x y w cuando esta arriba
% guardar acciones para llegar arriba
% agergar acciones - 4 una entre 2 y 1 y otra menor a 1
% explorar y eplotar arriba
% bajr velocidad
