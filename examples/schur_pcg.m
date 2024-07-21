% this file generates a random set of {A, B, b, Q, R, q, r} & x0 for LQR
% it then computes the Schur complement set (S, Pinv, gamma) of LQR in two ways 
% 1. traditional (as introduced in Brian's paper)
% 2. transformed (LDL on the diagonal blocks)

% two types of Schur set are saved to the data/ directory as txt files 
% 1. S.txt & P.txt & gamma.txt
% 2. Sdb.txt & Sob.txt & Pdb.txt & Pob.txt & gamma_tilde.txt

% pcg_solve.cu & pcg_block_solve.cu will read in txt files 
% and perform PCG separately

clc
close all
digits(16)
N = 100;
nx = 15;
nu = 1;

% generate a set of random {A_k, B_k}
A = cell(1, N-1);
B = cell(1, N-1);
for i=1:N
    A{i} = rand(nx, nx);
    B{i} = rand(nx, nu);
end

% generate a set of random positive definite {Q_k, R_k}
Q = cell(1, N);
R = cell(1, N-1);
for i=1:N-1
    T = unitaryMatrix(nx);
    Q{i} = T*diag(rand(nx,1))*T';
    R{i} = diag(rand(nu,1));
end
T = unitaryMatrix(nx);
Q{N} = T*diag(rand(nx,1))*T';

% compute S using {A_k, B_k, Q_k, R_k}
[D, O, S] = formKKTSchur(A, B, Q, R, N);
[D_P, O_P, P] = formPreconditionerSS(D, O, N, nx);
if ~exist('data', 'dir')
   mkdir('data')
end
writeMatrixToFile(D, O, N, nx, './data/S.txt');
writeMatrixToFile(D_P, O_P, N, nx, './data/P.txt');

% compute S using {\tilde{A}_k, \tilde{B}_k, \tilde{Q}_k, R_k}
% \tilde{A}_k = T_k * A_k * inv(T_{k-1})
% \tilde{B}_k = T_k * B_k
% \tilde{Q}_k = inv(T_k') * Q_k * inv(T_k)
% R_k remains the same
[D_tilde, O_tilde, S_tilde, T] = preprocessS(D, O, N, nx);
[D_P_tilde, O_P_tilde, P_tilde] = formPreconditionerSS(D_tilde, O_tilde, N, nx);
writeMatrixToFileDiagonal(D_tilde, O_tilde, N, nx, './data/S');
writeMatrixToFileDiagonal(D_P_tilde, O_P_tilde, N, nx, './data/P');

% generate random \gamma as RHS
gamma = rand(N*nx, 1);
writematrix(gamma, './data/gamma.txt');

% solve S * \lambda = \gamma
% lambda = S \ gamma;
pcg_max_iter = 1000;
tic
[lambda, I]= PCG(P, S, gamma, zeros(N*nx,1), 1e-8, pcg_max_iter);
toc

% prepare \tilde{\gamma} = T * \gamma
gamma_tilde = zeros(N*nx, 1);
for i=1:N
    gamma_tilde(1+(i-1)*nx:i*nx,1) = T{i}*gamma(1+(i-1)*nx:i*nx,1);
end
writematrix(gamma_tilde, './data/gamma_tilde.txt');

% solve \tilde{S} * \tilde{\lambda} = \tilde{\gamma}
% lambda_tilde = S_tilde \ gamma_tilde;
tic
[lambda_tilde, I_tilde]= PCG(P_tilde, S_tilde, gamma_tilde, zeros(N*nx,1), 1e-8, pcg_max_iter);
toc

% get back \lambda = T' * \tilde{\lambda}
lambda_new = zeros(N*nx, 1);
for i=1:N
    lambda_new(1+(i-1)*nx:i*nx,1) = T{i}'*lambda_tilde(1+(i-1)*nx:i*nx,1);
end

disp(['norm of lambda = ', num2str(norm(lambda))])
disp(['norm of lambda_tilde = ', num2str(norm(lambda_tilde))])

% check two lambda are equal 
disp(['norm of lambda - lambda_new = ', num2str(norm(lambda_new - lambda))])

% check condition numbers 
disp(['cond(P*S) = ', num2str(cond(P*S)), ', cond(P_tilde*S_tilde) = ', num2str(cond(P_tilde*S_tilde))])

disp(['PCG iterations for P*S = ', num2str(I), ', for P_tilde*S_tilde = ', num2str(I_tilde)])

function writeMatrixToFileDiagonal(D, O, N, nx, matrixname)
    % (D, O) forms a block tridiagonal matrix
    % each block is of size nx
    % elements of D are diagonals
    db = zeros(N*nx, 1);
    ob = zeros(N*2, nx*nx);

    db(1:nx) = diag(D{1});
    ob(2, :) = O{1}(:);
    for i=2:N-1
        db(1+(i-1)*nx:i*nx) = diag(D{i});

        offset = (i-1)*2;
        tmp = O{i-1}';
        ob(offset+1, :) = tmp(:);
        ob(offset+2, :) = O{i}(:);
    end
    db(end-nx+1:end) = diag(D{N});

    tmp = O{N-1}';
    ob(end-1, :) = tmp(:);

    filenamedb = [matrixname, 'db.txt'];
    writematrix(db, filenamedb);
    filenameob = [matrixname, 'ob.txt'];
    writematrix(ob, filenameob);
end

function writeMatrixToFile(D, O, N, nx, filename)
    % (D, O) forms a block tridiagonal matrix
    % each block is of size nx
    out = zeros(N*3, nx*nx);
    out(2, :) = D{1}(:);
    out(3, :) = O{1}(:);
    for i=2:N-1
        offset = (i-1)*3;
        tmp = O{i-1}';
        out(offset+1, :) = tmp(:);
        out(offset+2, :) = D{i}(:);
        out(offset+3, :) = O{i}(:);
    end
    tmp = O{N-1}';
    out(end-1, :) = D{N}(:);
    out(end-2, :) = tmp(:);
    writematrix(out, filename);
end

function U = unitaryMatrix(n) 
    % generate a random real matrix  
    X = rand(n)/sqrt(2); 
    % factorize the matrix 
    [Q,R] = qr(X); 
    R = diag(diag(R)./abs(diag(R))); 
    % unitary matrix 
    U = Q*R; 
end

function [D_pre, O_pre, S_pre, T] = preprocessS(D, O, N, nx)
    D_pre = cell(1,N);
    O_pre = cell(1,N-1);
    T = cell(1, N);

    [L1, D1] = ldl(D{1});
    T{1} = inv(L1);
    D_pre{1} = D1;

    for i=1:N-1
        [Li1, Di1] = ldl(D{i+1});
        T{i+1} = inv(Li1);
        D_pre{i+1} = Di1;
        O_pre{i} = T{i}*O{i}*T{i+1}';
    end

    S_pre = composeBlockDiagonalMatrix(D_pre, O_pre, N, nx);
end

function [D, O, S] = formKKTSchur(A, B, Q, R, N)
    D = cell(1, N);
    O = cell(1, N-1);

    nx = size(A{1}, 1);
    D{1} = inv(Q{1});
    for i=1:N-1
        D{i+1} = A{i}*inv(Q{i})*A{i}' + B{i}*inv(R{i})*B{i}' + inv(Q{i+1}); % theta{i}
        O{i} = -A{i}*inv(Q{i}); % phi{i}
        O{i} = O{i}';
    end

    S = composeBlockDiagonalMatrix(D, O, N, nx);
end

function [D_P, O_P, P] = formPreconditionerSS(D, O, N, nx)
    D_P = cell(1, N);
    O_P = cell(1, N-1); 

    for i=1:N-1
        D_P{i} = inv(D{i});
        O_P{i} = -inv(D{i})*O{i}*inv(D{i+1});
    end
    D_P{N} = inv(D{N});

    P = composeBlockDiagonalMatrix(D_P, O_P, N, nx);
end

function out = composeBlockDiagonalMatrix(D, O, N, nx)
    out = zeros(N*nx);
    out(1:nx, 1:2*nx) = [D{1}, O{1}];
    for i=2:N-1
        out((i-1)*nx+1:i*nx, (i-2)*nx+1:(i+1)*nx) = [O{i-1}', D{i}, O{i}];
    end
    out(end-nx+1:end, end-2*nx+1:end) = [O{N-1}', D{N}];
end

function [lambda, i] = PCG(Pinv, S, gamma, lambda_0, tol, max_iter)
    lambda = lambda_0;
    r = gamma - S*lambda;
    p = Pinv*r;
    r_tilde = p;
    nu = r'*r_tilde;
    for i=1:max_iter
       tmp = S*p;
       alpha = nu / (p'*tmp);
       r = r - alpha*tmp;
       lambda = lambda + alpha*p;
       r_tilde = Pinv*r;
       nu_prime = r'*r_tilde;
       if abs(nu_prime)<tol
          break
       end
       beta = nu_prime / nu;
       p = r_tilde + beta*p;
       nu = nu_prime;
    end
end