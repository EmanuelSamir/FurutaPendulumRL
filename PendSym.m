%% Dynamic plot with integration
clear all; close all; clc;

parameters = [300 100 750 500];
state = [pi-0.1 0];
T = 0;
dt = 0.05;
simS = [];
simS = [simS; state(1:2)];

for iter = 1:10000
    for i = 1:2
        k1 = Dynamics(state,T);
        k2 = Dynamics(state+dt/2*k1,T);
        k3 = Dynamics(state+dt/2*k2,T);
        k4 = Dynamics(state+dt*k3,T);

        state = state + dt/6*(k1 + 2*k2 + 2*k3 + k4);

        % Angle1 wrapped to 2pi
        if state(1)>pi
            state(1) = -pi + (state(1)-pi);
        elseif state(1)<-pi
            state(1) = pi - (-pi - state(1));
        end
    end
    
    simS = [simS; state(1:2)];
    
end

simulate1Pend(parameters, simS);