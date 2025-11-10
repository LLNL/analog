clear
close all
rng(123)
addpath ..\matlab-analog-sim

%% PARAMETERS

m = 40; % m^2 will be the A matrix size (Laplacian)
d = 4; % A = Lap + X X^T where X has d columns
d_sparsity = 0.10;
num_blocks = 20;
tol = 1e-12;
maxit = 200;

%% SETUP PROBLEM AND PRECONDITIONER

L = full(delsq(numgrid('S', m+2)));
n = size(L, 1);
X = sprand(n, d, d_sparsity);
A = L + X*X';
b = randn(n, 1);
b = b ./ norm(b);

eigval_L = eig(L); % for debugging purposes
eigval_A = eig(A); % for debugging purposes
% spy(A)

P_info3 = abj_setup(A, n);
P3 = @(u) abj_apply(P_info3, u);

P_info4 = abj_setup(A, num_blocks);
P4 = @(u) abj_apply(P_info4, u);

P_info5 = abj_setup(A, 1);
P5 = @(u) abj_apply(P_info5, u);

%% FGMRES

[x1, ~, ~, ~, resvec1] = gmres(A, b, [], tol, maxit, @(u) u, [], zeros(n, 1)); % GMRES (no precond.)
[x2, ~, ~, ~, resvec2] = gmres(A, b, [], tol, maxit, diag(diag(A)), [], zeros(n, 1)); % GMRES (Jacobi)
[x3, resvec3] = my_fgmres(A, b, tol, maxit, P3, zeros(n, 1)); % FGMRES (Jacobi)
[x4, resvec4] = my_fgmres(A, b, tol, maxit, P4, zeros(n, 1)); % FGMRES (intermediate #blocks)
[x5, resvec5] = my_fgmres(A, b, tol, maxit, P5, zeros(n, 1)); % FGMRES (one block, basically analog A^-1)

%% VISUALIZATION

f = figure;
semilogy(0:length(resvec1)-1, resvec1, '-k', 'LineWidth', 1.2)
hold on
semilogy(0:length(resvec2)-1, resvec2, '--k', 'LineWidth', 1.2)
semilogy(0:length(resvec3)-1, resvec3, '-r', 'LineWidth', 1.2)
semilogy(0:length(resvec4)-1, resvec4, '-b', 'LineWidth', 1.2)
semilogy(0:length(resvec5)-1, resvec5, '-g', 'LineWidth', 1.2)

xlabel('Iteration number')
ylabel('Relative residual norm')
title(sprintf('A = Lap + XX^T, n = %i', n))
legend('GMRES (iden.)', 'GMRES (Jacobi)', 'FGMRES (Jacobi)', sprintf('FGMRES (ABJ%i)', num_blocks), 'FGMRES (A^{-1})', 'Location', 'ne')
axis square
fontsize(14, "points")
exportgraphics(f, 'fgmres_abj_comparison.pdf')