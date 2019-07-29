function plot_furuta(th,al)
    %al = wrapToPi(al + pi);
    Lr = 0.0850;
    Lp = 0.1290;
    N = 30;
    % Forward Kinematics
    %th = pi/4;
    %al = pi/3;
    T_DH = @(th,d,a,al) [cos(th),    -cos(al)*sin(th),   sin(al)*sin(th),    a*cos(th);
                         sin(th),    cos(al)*cos(th),   -sin(al)*cos(th),    a*sin(th);
                         0,         sin(al)         ,   cos(al)         ,    d;
                         0,         0,              0                  ,1];
    T_0_1 = T_DH(th,0,0,pi/2);
    T_1_2 = T_DH(al,Lr,0,pi/2);
    T_2_3 = T_DH(0,Lp,0,0);
    % l1_x0 = 0;
    % l1_y0 = 0;
    % l1_z0 = 0;
    % l1_xf = l1_x0 + Lr*cos(th);
    % l1_yf = l1_y0 + Lr*sin(th);
    % l1_zf = l1_z0 + 0;
    % 
    % l2_x0 = l1_xf;
    % l2_y0 = l1_yf;
    % l2_z0 = l1_zf;
    % l2_xf = l1_xf



    L1 = T_0_1*T_1_2;
    L2 = T_0_1*T_1_2*T_2_3;

    l1_x0 = 0;
    l1_y0 = 0;
    l1_z0 = 0;
    l1_xf = L1(1,4);
    l1_yf = L1(2,4);
    l1_zf = L1(3,4);

    l2_x0 = l1_xf;
    l2_y0 = l1_yf;
    l2_z0 = l1_zf;
    l2_xf = L2(1,4);
    l2_yf = L2(2,4);
    l2_zf = L2(3,4);
    l1_xv = linspace(l1_x0,l1_xf,N);
    l1_yv = linspace(l1_y0,l1_yf,N);
    l1_zv = linspace(l1_z0,l1_zf,N);
    l2_xv = linspace(l2_x0,l2_xf,N);
    l2_yv = linspace(l2_y0,l2_yf,N);
    l2_zv = linspace(l2_z0,l2_zf,N);

    plot3(l1_xv,l1_yv,l1_zv, 'LineWidth',5, 'Color','red'); hold on;
    plot3(l1_x0,l1_y0,l1_z0, 'o','Color','red','LineWidth',5);
    plot3(l1_xf,l1_yf,l1_zf, 'o','Color','red','LineWidth',5);
    plot3(l2_xv,l2_yv,l2_zv,'Color','blue','LineWidth',5);
    plot3(l2_x0,l2_y0,l2_z0, 'o','Color','blue','LineWidth',5);
    plot3(l2_xf,l2_yf,l2_zf, 'o','Color','blue','LineWidth',5);hold off;
    axis equal;
    axis([-0.3 0.3 -0.3 0.3 -0.3 0.3]);
end