% DREM estimator for planar quadrotor model under a backup controller
%   a_x     = -K_{p,x}(x - x_b) - K_{d,x} xdot
%   a_z     = -K_{p,z}(z - z_b) - K_{d,z} zdot + g_D
%   theta_d = sat(atan(a_x/a_z))
%   F_b     = sat(m sqrt(a_x^2 + a_z^2))
%   M_b     = sat(J (K_{p,theta}(theta-theta_d) + K_{d,theta} thetadot))
%
% state:
%   X = [x, z, theta, xdot, zdot, thetadot]^T
%
% input:
%   u = [F, M]^T
%
% unknown parameters:
%   theta_p = [c_x, c_z, delta_g, delta_m, delta_J]^T

clear; close all; clc;

% sim
Tsim = 15.0;
dt   = 0.01;

[sim, par] = simulateBackupClosedLoop(Tsim, dt);
drem = runDREM(sim, par);

plotValidationFigure(sim, drem, par);
pe = computePEDiagnostic(sim, par);
plotPEDiagnostic(pe);

% closed-loop system
function [sim, par] = simulateBackupClosedLoop(Tsim, dt)

    % parameters from the quadrotor example (generalized bCBF paper)
    par.m  = 1.0;
    par.J  = 0.25;
    par.gD = 9.81;

    par.Fmax = 12.0;
    par.Mmax = 12.0;

    % generalized bCBF paper paper does not list the numerical backup
    % gains. I picked as

    par.Kpx = 0.25;
    par.Kdx = 1.30;
    par.Kpz = 1.20;
    par.Kdz = 2.40;
    par.Kpt = 4.00;
    par.Kdt = 4.00;

    % backup hover equilibrium x^* = [x_b, z_b, 0, 0, 0, 0]^T
    par.xb = 0.0;
    par.zb = 0.55;

    % saturation limits
    par.thetaMax = 0.45;

    % unknown parameters used in the true system
    % theta_p = [c_x, c_z, delta_g, delta_m, delta_J]^T.
    par.thetaTrue = [0.08; 0.12; 0.25; 0.11; -0.23];

    t = 0:dt:Tsim;
    Nt = numel(t);

    X  = zeros(6, Nt);
    U  = zeros(2, Nt);
    dX = zeros(6, Nt);
    thetaD = zeros(1, Nt);

    % X(:,1) = [-0.95; 1.55; 0.18; 0.0; -0.20; 0.0]; % not PE
    X(:,1) = [-0.95; 1.55; 0.35; 0.40; -0.35; 0.60];  % not PE

    for k = 1:Nt-1
        [U(:,k), thetaD(k)] = backupController(X(:,k), par);
        dX(:,k) = trueDynamics(X(:,k), par.thetaTrue, par);
        X(:,k+1) = rk4Step(X(:,k), dt, par.thetaTrue, par);
    end

    [U(:,Nt), thetaD(Nt)] = backupController(X(:,Nt), par);
    dX(:,Nt) = trueDynamics(X(:,Nt), par.thetaTrue, par);

    sim.t = t;
    sim.dt = dt;
    sim.X = X;
    sim.U = U;
    sim.dX = dX;
    sim.thetaD = thetaD;
end

% backup controller
function [u, theta_d] = backupController(X, par)
    % generalized bCBF paper's backup controller

    x  = X(1);
    z  = X(2);
    th = X(3);
    xd = X(4);
    zd = X(5);
    om = X(6);

    ax = -par.Kpx*(x - par.xb) - par.Kdx*xd;
    az = -par.Kpz*(z - par.zb) - par.Kdz*zd + par.gD;

    theta_d = sat(atan(ax/az), -par.thetaMax, par.thetaMax);

    Fb = sat(par.m*sqrt(ax^2 + az^2), 0.0, par.Fmax);

    Mb = sat( ...
        par.J*(par.Kpt*(th - theta_d) + par.Kdt*om), ...
        -par.Mmax, par.Mmax);

    u = [Fb; Mb];
end

% true dynamics of the system
function dX = trueDynamics(X, theta_p, par)
    % true uncertain planar quadrotor under the backup controller
    % nominal paper dynamics:
    %   xddot     = F sin(theta)/m
    %   zddot     = F cos(theta)/m - gD
    %   thetaddot = -M/J
    %
    %   with parametric uncertainty:
    %   xddot     = F sin(theta)/m - c_x xdot + delta_m F sin(theta)
    %   zddot     = F cos(theta)/m - gD - c_z zdot - delta_g ...
    %               + delta_m F cos(theta)
    %   thetaddot = -M/J - delta_J M

    c_x     = theta_p(1);
    c_z     = theta_p(2);
    delta_g = theta_p(3);
    delta_m = theta_p(4);
    delta_J = theta_p(5);

    [u, ~] = backupController(X, par);
    F = u(1);
    M = u(2);

    th = X(3);
    xd = X(4);
    zd = X(5);
    om = X(6);

    dX = [
        xd;
        zd;
        om;
        (1/par.m + delta_m)*F*sin(th) - c_x*xd;
        -par.gD - delta_g + (1/par.m + delta_m)*F*cos(th) - c_z*zd;
        -(1/par.J + delta_J)*M
    ];
end

% RK method
function Xnext = rk4Step(X, dt, theta_p, par)

    k1 = trueDynamics(X, theta_p, par);
    k2 = trueDynamics(X + 0.5*dt*k1, theta_p, par);
    k3 = trueDynamics(X + 0.5*dt*k2, theta_p, par);
    k4 = trueDynamics(X + dt*k3, theta_p, par);

    Xnext = X + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
end

% DREM
function drem = runDREM(sim, par)

    t  = sim.t;
    dt = sim.dt;
    Nt = numel(t);

    X  = sim.X;
    U  = sim.U;
    dX = sim.dX;

    Ntheta = 5;
    lambdaF = 1.0;

    Me = zeros(Ntheta, Ntheta);
    Ye = zeros(Ntheta, 1);

    chi = zeros(1, Nt);
    YD  = zeros(Ntheta, Nt);

    for k = 1:Nt
        [R, y] = regressionRows(X(:,k), U(:,k), dX(:,k), par);

        Me = Me + dt*(-lambdaF*Me + R'*R);
        Ye = Ye + dt*(-lambdaF*Ye + R'*y);

        chi(k) = det(Me);
        YD(:,k) = adjugateMatrix(Me)*Ye;
    end

    % Causal normalization of the scalar mixed regressor.
    % This keeps the DREM update numerically well scaled without using
    % future values of chi. Therefore, changing Tsim does not change the
    % estimator behavior over the common time interval.
    chiScale = zeros(1,Nt);
    chiScale(1) = max(abs(chi(1)), 1e-12);
    for k = 2:Nt
        chiScale(k) = max(chiScale(k-1), abs(chi(k)));
    end

    chiBar = chi./chiScale;
    YBar = zeros(size(YD));
    for k = 1:Nt
        YBar(:,k) = YD(:,k)/chiScale(k);
    end

    thetaLower = [-0.20; -0.20; -0.50; -0.30; -0.50];
    thetaUpper = [ 0.20;  0.20;  0.50;  0.30;  0.50];
    thetaHat0 = 0.5*(thetaLower + thetaUpper);
    rho0 = max(abs(thetaHat0 - thetaLower), abs(thetaUpper - thetaHat0));
    gamma = 0.8*ones(Ntheta,1);

    thetaHat = zeros(Ntheta, Nt);
    rho = zeros(Ntheta, Nt);

    thetaHat(:,1) = thetaHat0;
    rho(:,1) = rho0;

    for k = 1:Nt-1
        thetaHat(:,k+1) = thetaHat(:,k) + ...
            dt*gamma.*chiBar(k).*(YBar(:,k) - chiBar(k)*thetaHat(:,k));

        rho(:,k+1) = rho(:,k).*exp(-gamma*chiBar(k)^2*dt);
    end

    thetaErr = par.thetaTrue - thetaHat;
    ratio = abs(thetaErr)./(rho + 1e-14);

    mixedResidual = zeros(1,Nt);
    for k = 1:Nt
        mixedResidual(k) = max(abs(YBar(:,k) - chiBar(k)*par.thetaTrue));
    end

    drem.chi = chi;
    drem.chiBar = chiBar;
    drem.YBar = YBar;
    drem.thetaHat = thetaHat;
    drem.thetaErr = thetaErr;
    drem.rho = rho;
    drem.rho0 = rho0;
    drem.gamma = gamma;
    drem.mixedResidual = mixedResidual;
    drem.maxRatio = max(ratio, [], 'all');
end

% 
function [R, y] = regressionRows(X, U, dX, par)

    th = X(3);
    xd = X(4);
    zd = X(5);

    F = U(1);
    M = U(2);

    % y = R theta_p, theta_p = [c_x, c_z, delta_g, delta_m, delta_J]^T.
    R = [
        -xd,  0,   0, F*sin(th),  0;
          0, -zd, -1, F*cos(th),  0;
          0,  0,   0,         0, -M
    ];

    y = [
        dX(4) - F*sin(th)/par.m;
        dX(5) - (-par.gD + F*cos(th)/par.m);
        dX(6) - (-M/par.J)
    ];
end

% PE condition diagnostic
function pe = computePEDiagnostic(sim, par)

    t  = sim.t;
    dt = sim.dt;
    Nt = numel(t);

    X  = sim.X;
    U  = sim.U;
    dX = sim.dX;

    Ntheta = 5;
    Tw = 1.5;
    Nw = round(Tw/dt);
    validN = Nt - Nw;

    eigVals = zeros(Ntheta, validN);
    tWin = t(1:validN);

    for k = 1:validN
        W = zeros(Ntheta, Ntheta);
        for j = k:(k+Nw)
            [R, ~] = regressionRows(X(:,j), U(:,j), dX(:,j), par);
            W = W + R'*R*dt;
        end
        eigVals(:,k) = sort(eig(W), 'ascend');
    end

    gammaPE = eigVals(1,:);

    fprintf('max gamma_PE(t) = %.3e\n', max(gammaPE));
    fprintf('min gamma_PE(t) = %.3e\n', min(gammaPE));
    fprintf('final sliding-window eigenvalues:\n');
    disp(eigVals(:,end).');

    pe.tWin = tWin;
    pe.Tw = Tw;
    pe.eigVals = eigVals;
    pe.gammaPE = gammaPE;
end

% plot PE condition diagnostic
function plotPEDiagnostic(pe)

    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultAxesFontSize', 13);
    set(groot, 'defaultLineLineWidth', 2);

    fig = figure('Color','w', 'Units','inches', 'Position',[1, 1, 7.35, 2.55]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

    % PE indicator
    ax = nexttile(tl, 1);
    hold(ax, 'on'); box(ax, 'on');

    plot(ax, pe.tWin, pe.gammaPE, 'DisplayName', '$\gamma_{\rm PE}(t)$');
    yline(ax, 0, '--', 'LineWidth', 1.0, 'DisplayName', '$0$');

    xlabel(ax, '$t$ [s]');
    ylabel(ax, '$\lambda_{\min}(W_T(t))$');
    xlim(ax, [0, pe.tWin(end)]);

    ymax = max(pe.gammaPE);
    if ymax <= 0
        ymax = 1;
    end
    ylim(ax, [-0.02*ymax, 1.05*ymax]);

    legend(ax, 'Location','northeast', 'Box','on');

    % eigenvalues of the sliding-window information matrix
    ax = nexttile(tl, 2);
    hold(ax, 'on'); box(ax, 'on');

    for i = 1:5
        plot(ax, pe.tWin, pe.eigVals(6-i,:), 'DisplayName', ['$\lambda_', num2str(i), '$']);
    end

    xlabel(ax, '$t$ [s]');
    ylabel(ax, 'eigenvalues of $W_T(t)$');
    xlim(ax, [0, pe.tWin(end)]);

    legend(ax, 'Location','northeast', 'NumColumns',2, 'Box','on');

    exportgraphics(fig, 'drem_quadrotor_pe.pdf', 'ContentType','vector');
end

% plots
function plotValidationFigure(sim, drem, par)

    t = sim.t;
    X = sim.X;
    U = sim.U;

    thetaHat = drem.thetaHat;
    thetaErr = drem.thetaErr;
    rho = drem.rho;
    rho0 = drem.rho0;

    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultAxesFontSize', 13);
    set(groot, 'defaultLineLineWidth', 2);

    fig = figure('Color','w', 'Units','inches', 'Position',[1, 1, 7.35, 4.95]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact', 'Padding','compact');

    colors = lines(5);

    % closed-loop backup trajectory
    ax = nexttile(tl, 1);
    hold(ax, 'on'); box(ax, 'on');

    plot(ax, X(1,:), X(2,:), 'LineWidth', 2.2, 'DisplayName', 'backup trajectory');
    plot(ax, X(1,1), X(2,1), 'o', 'MarkerSize', 5.0, 'LineStyle','none', 'DisplayName', '$p(0)$');
    plot(ax, par.xb, par.zb, '*', 'MarkerSize', 9.0, 'LineStyle','none', 'DisplayName', '$p_{\rm b}$');

    xL = -1.25; xR = 1.05; zL = 0.0; zT = 1.85;
    plot(ax, [xL xL], [zL zT], '--', 'LineWidth', 1.0, 'DisplayName', 'constraints');
    plot(ax, [xR xR], [zL zT], '--', 'LineWidth', 1.0, 'HandleVisibility','off');
    plot(ax, [xL xR], [zL zL], '--', 'LineWidth', 1.0, 'HandleVisibility','off');

    xlabel(ax, '$x$ [m]');
    ylabel(ax, '$z$ [m]');
    xlim(ax, [-1.35, 1.15]);
    ylim(ax, [-0.08, 1.95]);
    axis(ax, 'equal');

    legend(ax, 'Location','northeast', 'NumColumns',2, 'Box','on');

    % inputs
    ax = nexttile(tl, 2);
    hold(ax, 'on'); box(ax, 'on');

    plot(ax, t, U(1,:), 'DisplayName', '$F_b$');
    plot(ax, t, U(2,:), 'DisplayName', '$M_b$');
    yline(ax, par.Fmax, '--', 'LineWidth', 1.0, 'DisplayName', 'bounds');
    yline(ax, par.Mmax, '--', 'LineWidth', 1.0, 'HandleVisibility','off');
    yline(ax, -par.Mmax, '--', 'LineWidth', 1.0, 'HandleVisibility','off');

    xlabel(ax, '$t$ [s]');
    ylabel(ax, 'input');
    xlim(ax, [0, t(end)]);

    lgd = legend(ax, 'Location','north', 'NumColumns',3, 'Box','on');
    lgd.Position(2) = lgd.Position(2) - 0.045;

    % parameter estimates
    ax = nexttile(tl, 3);
    hold(ax, 'on'); box(ax, 'on');

    labelsHat = {'$\hat c_x$', '$\hat c_z$', '$\hat \delta_g$', '$\hat \delta_m$', '$\hat \delta_J$'};

    for i = 1:5
        plot(ax, t, thetaHat(i,:), 'Color', colors(i,:), 'DisplayName', labelsHat{i});
        yline(ax, par.thetaTrue(i), '--', 'Color', colors(i,:), 'LineWidth', 1.2, 'HandleVisibility','off');
    end

    plot(ax, nan, nan, 'k--', 'LineWidth', 1.2, 'DisplayName', 'true values');

    xlabel(ax, '$t$ [s]');
    ylabel(ax, 'parameter estimate');
    xlim(ax, [0, t(end)]);

    lgd = legend(ax, 'Location','south', 'NumColumns',3, 'Box','on');
    lgd.Position(1) = lgd.Position(1) + 0.055;
    lgd.Position(2) = lgd.Position(2) + 0.055;

    % element-wise errors and bounds
    ax = nexttile(tl, 4);
    hold(ax, 'on'); box(ax, 'on');

    labelsErr = {'$|\tilde c_x|$', '$|\tilde c_z|$', '$|\tilde \delta_g|$', '$|\tilde \delta_m|$', '$|\tilde \delta_J|$'};

    for i = 1:5
        plot(ax, t, abs(thetaErr(i,:)), 'Color', colors(i,:), 'DisplayName', labelsErr{i});
        plot(ax, t, rho(i,:), '--', 'Color', colors(i,:), 'LineWidth', 1.2, 'HandleVisibility','off');
    end

    plot(ax, nan, nan, 'k--', 'LineWidth', 1.2, 'DisplayName', '$\rho_i$');

    xlabel(ax, '$t$ [s]');
    ylabel(ax, 'error and bound');
    xlim(ax, [0, t(end)]);
    ylim(ax, [0, 1.08*max(rho0)]);

    legend(ax, 'Location','north', 'NumColumns',3, 'Box','on');

    exportgraphics(fig, 'drem_quadrotor.pdf', 'ContentType','vector');
end


% adjugate matrix
function Aadj = adjugateMatrix(A)

    n = size(A,1);
    Aadj = zeros(n,n);

    for i = 1:n
        for j = 1:n
            Aij = A;
            Aij(j,:) = [];
            Aij(:,i) = [];
            Aadj(i,j) = (-1)^(i+j)*det(Aij);
        end
    end
end

% saturation
function y = sat(x, xmin, xmax)
    y = min(max(x, xmin), xmax);
end
