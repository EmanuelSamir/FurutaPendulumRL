function [ dotx ] = furutaNonLinealModel( x, input )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

theta = x(1);
alpha = x(2);
dottheta = x(3);
dotalpha = x(4);    
u = input; % input voltage Vm

% Plant parameters
% Resistance
Rm = 8.4;
% Back-emf constant (V-s/rad)
km = 0.042;
% Mass (kg)
Mr = 0.095;
% Total length (m)
Lr = 0.085;
% Moment of inertia about pivot (kg-m^2)
Jr = Mr*Lr^2/12;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
Dr = 0.0015;
%
% Pendulum Link
% Mass (kg)
mp = 0.024;
% Total length (m)
Lp = 0.129;
% Moment of inertia about pivot (kg-m^2)
Jp = mp*Lp^2/12;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
Dp = 0.0005;
% Gravity Constant
g = 9.81;

% D, C, G and H matrices parameters
d11 = Rm/km*(mp*Lr^2 + 1/4*mp*Lp^2*cos(alpha)^2 + Jr);
d12 = Rm/km*(-1/2*mp*Lp*Lr*cos(alpha));
d21 = (1/2*mp*Lp*Lr*cos(alpha));
d22 = (Jp + 1/4*mp*Lp^2);

c11 = Rm/km*(1/2*mp*Lp^2*sin(alpha)*cos(alpha)*dotalpha + km^2/Rm + Dr);
c12 = Rm/km*(1/2*mp*Lp*Lr*sin(alpha)*dotalpha);
c21 = -1/4*mp*Lp^2*cos(alpha)*sin(alpha)*dottheta;
c22 = Dp;

g1 = 0;
g2 = 1/2*mp*Lp*g*sin(alpha);

h1 = 1;
h2 = 0;

D = [d11 d12;
     d21 d22];

C = [c11 c12;
     c21 c22];

G = [g1; g2];

H = [h1; h2];

A = zeros(4,4);
A(1:2,3:4) = eye(2,2);
A(3:4,3:4) = -D\C;

B1 = [0; 0; D\H];

B2 = [0; 0; -D\G];

dotx = A*x + B1*u + B2;

end

