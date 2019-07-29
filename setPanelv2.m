function [joint, map, map2, pathmap, pathmap2] = setPanelv2( parameters, fontsize, width, height )
%function joint = setPanel( parameters )
%   This function sets the inital configurations for the simulation panel.
%   And creates a figure with them. The users might want to set the a, b, c
%   and d parameters properly for the figure to fit their screens as
%   wished.
%
% Inputs:
%   parameters = [a b c d] 1x4 vector
%   fontsize = fontsize of the graph's labels
%   width = length of the angle state
%   height = length of the rate state
% where:
%   a = figure distance from left of the screen
%   b = figure distance from bottom of the screen
%   c = figure width
%   d = figure height
%
% Outputs:
%   joint = properties for the joint plot
%   map = graph properties
%   pathmap = properties for the marker that travels through graph

%-- Panel configuration --%
panel = figure;
panel.Position = parameters;
panel.Color = [1 1 1];

%-- Plot the pendulum joint --%
subplot(2,4,[1,5])
hold on
% Axis for the pendulum animation
joint = plot(0,0,'b','LineWidth',10);
axPend = joint.Parent;
axPend.XTick = []; % No axis stuff to see
axPend.YTick = [];
% axPend.Position = [e f g h]
% e = distance from left of the figure
% f = distance from bottom of the figure
% g = plot width
% h = plot height
axPend.Position = [0.01 0.4 0.3 0.5];
axPend.Visible = 'off';
axPend.Clipping = 'off';
axis equal
axis([-1.2679 1.2679 -1 1]); % [xmin xmax ymin ymax]
plot(0.001,0,'.k','MarkerSize',50); % Pendulum joint
hold off

%-- Plot q1 angle and rate graph --%
subplot(2,4,[2:4]);
colormap('gray');
hold on
map = imagesc(zeros(height*width,1));
axMap = map.Parent;
axMap.XTickLabels = {'-pi' '0' 'pi'}; % Label x axis
axMap.XTick = [1 floor(width/2) width]; % Limit x axis labels
axMap.YTickLabels = {'-pi' '0' 'pi'}; % Label y axis
axMap.YTick = [1 floor(height/2) height]; % Limit y axis labels
axMap.XLabel.String = 'q1 Angle (rad)'; % Name x axis
axMap.YLabel.String = 'q1 Angular rate (rad/s)'; % Name y axis
axMap.Visible = 'on'; % Make sure axis' visibility is turned on
axMap.Color = [0.3 0.3 0.5];
axMap.XLim = [1 width]; % Resize x axis
axMap.YLim = [1 height]; % Resize y axis
axMap.Box = 'off';
axMap.FontSize = fontsize; % Change axis' fontsize
caxis([-37,-0.0005]) % Set minimum and maximum color limits
pathmap = plot(NaN,NaN,'.g','MarkerSize',30); % The green marker that travels through the state map to match the pendulum animation
hold off

%-- Plot q2 angle and rate graph --%
subplot(2,4,[6:8]);
colormap('gray');
hold on
map2 = imagesc(zeros(height*width,1));
axMap = map2.Parent;
axMap.XTickLabels = {'-pi' '0' 'pi'}; % Label x axis
axMap.XTick = [1 floor(width/2) width]; % Limit x axis labels
axMap.YTickLabels = {'-pi' '0' 'pi'}; % Label y axis
axMap.YTick = [1 floor(height/2) height]; % Limit y axis labels
axMap.XLabel.String = 'q2 Angle (rad)'; % Name x axis
axMap.YLabel.String = 'q2 Angular rate (rad/s)'; % Name y axis
axMap.Visible = 'on'; % Make sure axis' visibility is turned on
axMap.Color = [0.3 0.3 0.5];
axMap.XLim = [1 width]; % Resize x axis
axMap.YLim = [1 height]; % Resize y axis
axMap.Box = 'off';
axMap.FontSize = fontsize; % Change axis' fontsize
caxis([-37,-0.0005]) % Set minimum and maximum color limits
pathmap2 = plot(NaN,NaN,'.g','MarkerSize',30); % The green marker that travels through the state map to match the pendulum animation
hold off

end