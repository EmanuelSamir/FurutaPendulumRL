function simulate1Pend(parameters, angleStates)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

panel = figure;
panel.Position = parameters;
panel.Color = [1 1 1];
hold on;

% Plot first bar
bar = plot(0,0.3,'b','LineWidth',10); % Invisible at the beginning

% Graph configurations
axPend = bar.Parent;
axPend.XTick = []; % No axis stuff to see
axPend.YTick = [];
axPend.Position = [0.24 0.18 0.5 0.8];
axPend.Visible = 'off'; % Turn off border
axPend.Clipping = 'off';
axis equal
axis([-2 2 -2 2]); % [xmin xmax ymin ymax]

% Plot first joint
plot(0,0,'.k','MarkerSize',50);

for i = 1:size(angleStates,1)
    % Move bar
    set(bar,'XData',[ 0 sin(angleStates(i,1)) ]);
    set(bar,'YData',[ 0 -cos(angleStates(i,1)) ]);
    pause(0.1);   
end

end

