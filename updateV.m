function [ newx, newy ] = updateV( Q, map, width, height, index)
%function [ newx, newy ] = untitled( Q, width, height, index)
%   This function updates the value function matrix and outputs the
%   coordinates of the values that changed.
%
% Inputs:
%   Q = action-value function
%   map = graph properties
%   width = length of the angle state
%   height = length of the rate staet
%   index = position of the current state
%
% Outputs:
%   newx = x coordinate of the changed value
%   newy = y coordinate of the changed value

[newy,newx] = ind2sub([height,width],index); % Find the 2d index of the 1d state index we found above
% The heat map of best Q values
V = max(Q,[],2); % Best estimated value for all actions at each state.
fullV = reshape(V,[height,width]); % Make into 2D for plotting instead of a vector.
set(map,'CData',fullV);

end

