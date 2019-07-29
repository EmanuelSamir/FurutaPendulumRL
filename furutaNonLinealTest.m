%% Script for non lineal model testing
EPIS_MAX = 1000;
f = 100;                                 % Frequency
dt = 1/f;                                % delta T. 
Tend = 5;                                % Final Time
tspan = 0:dt:Tend;                       % Time Span
th0 = 0; dth0 = 0; al0 = 0; dal0 = 0;    % Initial Conditions
x0 = [th0; al0; dth0; dal0];             % Initial State
u = 0;                                   % Initial Input
x = x0 + dt*furutaNonLinealModel(x0, u); % State First Step

C = eye(2,4);
D = zeros(2,1);

% To record data
X = zeros(length(x0),length(tspan));
Y = zeros(2,length(tspan));
k = 0;

action_set = [-3, -2, -1, 0, 1, 2, 3];

f = figure(1);
for t = tspan
    k = k+1;
    
    u = 3*sin(t*12);
    
    % System Saving and Show
    Y(:,k) = C*x + D*u;
    x = x + dt*furutaNonLinealModel(x, u);
    X(:,k) = x;
    plot_furuta(x(1), x(2));
    pause(dt);
end