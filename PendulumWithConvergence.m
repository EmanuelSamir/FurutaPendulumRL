function PendulumWithConvergence
%% Example reinforcement learning - Q-learning code
% Learn a control policy to optimally swing a pendulum from vertical down,
% to vertical up with torque limits and (potentially) noise. Both the
% pendulum and the policy are animated as the process is going. The
% difference from dynamic programming, for instance, is that the policy is
% learned only by doing forward simulation. No knowledge of the dynamics is
% used to make the policy.
%   
% Play around with the learning settings below. I'm sure they could be
% improved greatly!
%
%   Video: https://www.youtube.com/watch?v=YLAWnYAsai8
%
%   Matthew Sheen, 2015
%
% Edited by
%
% Alessio Ghio
%
% Edit:
%   Edited from "PendulumWithFunctions.m" (check description).
%   In this version, the algorithm does not iterate a set number of
%   times, but until it converges to a defined parameter theta. This
%   modification makes reference to Sutton's value iteration pseudo code.

clear all; close all; clc;

%% Determine initial parameters
theta = 1e-4; % Precision parameter
delta = 1; % convergence parameter
maxiter = 1500; % Set number of iterations because there are undefined states
gamma = 0.9; % Discount rate
Rfunc = @(x,xdot)(-(abs(x)).^2 + -0.25*(abs(xdot)).^2); % Reward function
state1 = [pi, 0]; % Initial position
epsilon = 0.5; % Probability of picking random action vs estimated best action
epsilonDecay = 0.98; % Decay factor per iteration.
TLim = 5; % Torque
actions = [-TLim 0 TLim]; % Only 3 options, Full blast one way, the other way, and off
dt = 0.05; % Timestep of integration. Each substep lasts this long
episodes = 1;
successRate = 1;
learnRate = 0.99;
deltaFlag = 0;

%% Set up simulation Panel
parameters = [200 170 900 450];
fontsize = 11;
%-- Set possible states for simulation --%
% Angle state
x1 = -pi:0.05:pi;
% Angular rate state
x2 = -pi:0.1:pi;
% Posible combination of discretized states
states=zeros(length(x1)*length(x2),2);
idx=1;
for j=1:length(x1)
    for k = 1:length(x2)
        states(idx,1)=x1(j);
        states(idx,2)=x2(k);
        idx=idx+1;
    end
end

V = zeros(size(states,1),1);
% Initialize the "cost" of a given state to be quadratic error from the 
% goal state. Note the signs mean that -angle with +velocity is better than
% -angle with -velocity
R = Rfunc(states(:,1),states(:,2));
Q = repmat(R,[1,3]);

% Initial coordinates
simC = [];
simS = state1;
%-----------------------------------%
[joint, map, pathmap] = setPanel(parameters, fontsize, length(x1), length(x2));
simPanel = zeros(length(x2), length(x1));

%% while loop

while(delta>theta)
    % Reset delta
    delta = 0;
    % Reset initial state
    state1 = [pi/2, 0];
    % Reset simulation matrices
    simS = [];
    simC = [];
    
    for iter = 1:maxiter
        v_temp = V;
        
        %-- Pick an action --%
        % Interpolate the state within our discretization
        [~,sIdx] = min(sum((states - repmat(state1,[size(states,1),1])).^2,2));
        
        % Pick according to the Q-matrix it's the last episode or we succeed with the rand()>epsilon check. 
        % Fail the check if our action doesn't succeed (i.e. simulating noise)
        if (rand()>epsilon) && rand()<=successRate 
            [~,aIdx] = max(Q(sIdx,:)); % Pick the action the Q matrix thinks is best!
        else
            aIdx = randi(length(actions),1); % Random action!
        end
        
        T = actions(aIdx);
        %--------------------------------------------------------%
        
        %-- Update state --%
        % Step the dynamics forward with our new action choice
        % RK4 Loop - Numerical integration
        for i = 1:2
            k1 = Dynamics(state1,T);
            k2 = Dynamics(state1+dt/2*k1,T);
            k3 = Dynamics(state1+dt/2*k2,T);
            k4 = Dynamics(state1+dt*k3,T);
            
            state2 = state1 + dt/6*(k1 + 2*k2 + 2*k3 + k4);
            % All states wrapped to 2pi
            if state2(1)>pi
                state2(1) = -pi + (state2(1)-pi);
            elseif state2(1)<-pi
                state2(1) = pi - (-pi - state2(1));
            end
        end
        
        state1 = state2;
        simS = [simS; state1];
        %-------------------------------------------------%
        
        %-- Update V and Q --%
        % End condition for an episode
        if norm(state2)<0.01 % If we've reached upright with no velocity (within some margin), end this episode
            success = true;
            bonus = 100; % Give a bonus for getting there
        else
            bonus = 0;
            success = false;
        end
        
        % Interpolate again to find the new state the system is closest to
        [~,snewIdx] = min(sum((states - repmat(state1,[size(states,1),1])).^2,2));
        
        % Bellman equation
        Q(sIdx,aIdx) = Q(sIdx,aIdx) + learnRate * ( R(snewIdx) + gamma*max(Q(snewIdx,:)) - Q(sIdx,aIdx) + bonus );
        % newQ = oldQ + learnRate*( Rfunc(state) + gamma*(V) - Qold + bonus )
        V = max(Q,[],2); % Best estimated value for all actions at each state.
        %-------------------------------------------------------%
        
        % Decay the odds of picking a random action vs picking the
        % estimated "best" action
        epsilon = epsilon*epsilonDecay;
        
        %-- Simulation --%
        % Find the 2d index of the 1d state index we found above
        [newx, newy] = updateV(Q, map, length(x1), length(x2), snewIdx);
        coords = [newx newy];
        simC = [simC; coords];
        %-----------------------------------------------------%
        
        delta = max(delta, max(abs(v_temp - V)));
        if success
            break;
        end
        if (delta<theta)
            deltaFlag = 1;
        end
    end % For loop
    
    if success
        disp(['Objective reached on episode nº ', int2str(episodes)])
        disp('Continue...')
        if deltaFlag
            delta = -inf;
        end

    else
        disp(['Didn´t reach the objective on episode nº ', int2str(episodes)])
        pause(0.00000001)% Pause needed for printing messages
        deltaFlag = 0;
    end
    
    episodes = episodes + 1;
    
end % While Loop

disp(['Converged with ', int2str(episodes), ' episodes']);
disp('Running last Simulation...');
updatePanel(joint, pathmap, simC, simS)

end % Reinforcement learning function

function zdot = Dynamics(state,T)
% Pendulum with motor at the joint dynamics. 
% IN - [angle,rate] & torque.
% OUT - [rate,accel]
g = 9.8;
L = 1;
state = state';
zdot = [state(2) g/L*sin(state(1))+T];
end
