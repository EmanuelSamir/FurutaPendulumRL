clearvars; close all; clc;
%% 
path = './PendulumData/';

alpha = csvread(strcat(path,'SwingUpData1/','alpha'));
alphadot = csvread(strcat(path,'SwingUpData1/','alphadot'));
theta = csvread(strcat(path,'SwingUpData1/','theta'));
thetadot = csvread(strcat(path,'SwingUpData1/','thetadot'));
Vin = csvread(strcat(path,'SwingUpData1/','Vin'));

dt = 0.002;                                % delta T. 
Tend = 10;                                % Final Time
tspan = 0:dt:Tend;                       % Time Span
x0 = [theta(1,2); wrapToPi(alpha(1,2) + pi); thetadot(1,2); alphadot(1,2) ];             % Initial State
u = Vin(1,2);                                   % Initial Input
x = x0 + dt*furutaNonLinealModel(x0, u); % State First Step
%%
C = eye(2,4);
D = zeros(2,1);

% To record data
X = zeros(length(x0),length(tspan));
Y = zeros(2,length(tspan));
k = 0;

f = figure(1);
for t = tspan
    if mod(k,1000) == 0
        disp(['Time: ', num2str(t)]);
    end
    k = k+1;
    u = -Vin(k,2);
    % System Saving and Show
    Y(:,k) = C*x + D*u;
    x = x + dt*furutaNonLinealModel(x, u);
    X(:,k) = x;
    plot_furuta(x(1), x(2));
    pause(dt/100);
end

%% Derivating theta and alpha
dtheta = [];
for i= 1:(length(theta(:,2))-1)
    if sign(theta(i,2))*sign(theta(i+1,2)) ~= 1
        dtheta = [dtheta, sign(theta(i+1,2))/dt*(abs(theta(i+1,2)) - abs(theta(i,2)))];
    end
    dtheta = [dtheta, (theta(i+1,2) - theta(i,2))/dt];  
end

%% Plotting Real data and Simulation Data
f2 = figure(3);
subplot(2,2,1); plot(theta(:,1),theta(:,2)); title('theta');hold on;
plot(tspan,X(1,:)); hold off; legend('real', 'simulation');
subplot(2,2,2); title('\alpha')
plot(alpha(:,1),wrapToPi(alpha(:,2)+ pi) ); hold on;
plot(tspan,X(2,:)); hold off;legend('real', 'simulation');

subplot(2,2,3); title('\dot\theta')
plot(thetadot(:,1),thetadot(:,2)); hold on;
plot(tspan,X(3,:)); hold off;legend('real', 'simulation');
subplot(2,2,4); 
plot(alphadot(:,1),alphadot(:,2)); hold on;
plot(tspan,X(4,:)); hold off;legend('real', 'simulation');

%% Save Data
csvwrite('./SwingUpData1_sim/alpha',X(1,:));
csvwrite('./SwingUpData1_sim/theta',X(2,:));
csvwrite('./SwingUpData1_sim/alphadot',X(3,:));
csvwrite('./SwingUpData1_sim/thetadot',X(4,:));
