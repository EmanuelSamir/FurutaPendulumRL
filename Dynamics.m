function zdot = Dynamics(state,T)
% Pendulum with motor at the joint dynamics. 
% IN - [angle,rate] & torque.
% OUT - [rate,accel]
g = 9.8;
L = 1;
m = 1;
b = 0.01;
state = state';
% zdot = [state(2) -g/L*sin(state(1))+T];
zdot = [state(2) -g/L*sin(state(1))+T/m/L/L-b*state(2)/m/L/L];
end