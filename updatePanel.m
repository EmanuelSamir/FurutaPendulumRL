function updatePanel(joint, pathmap, simC, simS)
%function updatePanel(joint, pathmap, state, sim)
%   This function simulates the pendulum dynamics and shows a cursor that
%   moves through the value function graph.
%
% Inputs:
%   joint = pendulum plot properties
%   pathmap = cursor plot properties
%   simC = matrix with coordinates for simulation
%   simS = matrix with states for simulation
%
% Outputs:
%   newx = x coordinate of the changed value
%   newy = y coordinate of the changed value

for i = 1:size(simC,1)
    % Pendulum state:
    set(joint,'XData',[0 -sin(simS(i,1))]);
    set(joint,'YData',[0 cos(simS(i,1))]);
    % Green tracer point:
    set(pathmap,'XData',simC(i,1));
    set(pathmap,'YData',simC(i,2));
    pause(0.03);
end % For loop

drawnow;

end % Function

