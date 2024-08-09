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
N = 20;
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

if ~exist('data', 'dir')
   mkdir('data')
else
   delete('./data/I_H*')
end

% compute S using {A_k, B_k, Q_k, R_k}
[D, O, S] = formKKTSchur(A, B, Q, R, N);
writeBlkTriDiagSymMatrixToFile(D, O, N, nx, './data/S.txt');

[D_P, O_P, P_p0s3] = formPreconditionerSS(D, O, N, nx);
writeBlkTriDiagSymMatrixToFile(D_P, O_P, N, nx, './data/P.txt');

[D_H, O_up2, O_down2, H] = formPolyPreconditionerH(D, O, N, nx);
writeBlkPentaDiagMatrixToFile(D_H, O_up2, O_down2, N, nx, './data/H.txt')
P_p1s3 = (eye(N*nx) + H) * P_p0s3;

% compute S using {\tilde{A}_k, \tilde{B}_k, \tilde{Q}_k, R_k}
% \tilde{A}_k = T_k * A_k * inv(T_{k-1})
% \tilde{B}_k = T_k * B_k
% \tilde{Q}_k = inv(T_k') * Q_k * inv(T_k)
% R_k remains the same
[D_til, O_til, S_til, T] = preprocessS(D, O, N, nx);
writeMatrixToFileDiagonal(D_til, O_til, N, nx, './data/S');

[D_P_til, O_P_til, P_p0s3_til] = formPreconditionerSS(D_til, O_til, N, nx);
writeMatrixToFileDiagonal(D_P_til, O_P_til, N, nx, './data/P');

[D_H_til, O_up2_til, O_down2_til, H_til] = formPolyPreconditionerH(D_til, O_til, N, nx);
writeBlkPentaDiagMatrixToFile(D_H_til, O_up2_til, O_down2_til, N, nx, './data/H_tilde.txt')
P_p1s3_til = (eye(N*nx) + H_til) * P_p0s3_til;

% generate random \gamma as RHS
gamma = rand(N*nx, 1);
writematrix(gamma, './data/gamma.txt');

% solve S * \lambda = \gamma
% lambda = S \ gamma;
pcg_max_iter = 1000;
[lambda_p0s3, I_p0s3]= PCG(P_p0s3, eye(N*nx), S, gamma, zeros(N*nx,1), 1e-8, pcg_max_iter);
[lambda_p1s3, I_p1s3]= PCG(P_p0s3, eye(N*nx)+H, S, gamma, zeros(N*nx,1), 1e-8, pcg_max_iter);

% prepare \tilde{\gamma} = T * \gamma
gamma_til = zeros(N*nx, 1);
for i=1:N
    gamma_til(1+(i-1)*nx:i*nx,1) = T{i}*gamma(1+(i-1)*nx:i*nx,1);
end
writematrix(gamma_til, './data/gamma_tilde.txt');

% solve \tilde{S} * \tilde{\lambda} = \tilde{\gamma}
% lambda_tilde = S_tilde \ gamma_tilde;
[lambda_p0s3_til, I_p0s3_til]= PCG(P_p0s3_til, eye(N*nx), S_til, gamma_til, zeros(N*nx,1), 1e-8, pcg_max_iter);
[lambda_p1s3_til, I_p1s3_til]= PCG(P_p0s3_til, eye(N*nx)+H_til, S_til, gamma_til, zeros(N*nx,1), 1e-8, pcg_max_iter);

% get back \lambda = T' * \tilde{\lambda}
lambda_new = zeros(N*nx, 1);
for i=1:N
    lambda_new(1+(i-1)*nx:i*nx,1) = T{i}'*lambda_p1s3_til(1+(i-1)*nx:i*nx,1);
end

disp(['norm of lambda p0s3 = ', num2str(norm(lambda_p0s3))])
disp(['norm of lambda p1s3 = ', num2str(norm(lambda_p1s3))])
disp(['norm of lambda_tilde p0s3 = ', num2str(norm(lambda_p0s3_til))])
disp(['norm of lambda_tilde p1s3 = ', num2str(norm(lambda_p1s3_til))])

% check two lambda are equal 
disp(['norm of lambda - lambda_new = ', num2str(norm(lambda_new - lambda_p1s3))])

% check condition numbers 
disp(['cond(P*S) p0s3 = ', num2str(cond(P_p0s3*S)), ', cond(P_til*S_til) p0s3 = ', num2str(cond(P_p0s3_til*S_til))])
disp(['cond(P*S) p1s3 = ', num2str(cond(P_p1s3*S)), ', cond(P_til*S_til) p1s3 = ', num2str(cond(P_p1s3_til*S_til))])

disp(['PCG iterations for P*S p0s3 = ', num2str(I_p0s3), ', for P_til*S_til p0s3 = ', num2str(I_p0s3_til)])
disp(['PCG iterations for P*S p1s3 = ', num2str(I_p1s3), ', for P_til*S_til p1s3 = ', num2str(I_p1s3_til)])

alpha = 1:0.5:5;
for i=1:length(alpha)
    I_H = eye(N*nx) + alpha(i)*H;
    [~, I_p1s3_a]= PCG(P_p0s3, I_H, S, gamma, zeros(N*nx,1), 1e-8, pcg_max_iter);
    
    I_H_til = eye(N*nx) + alpha(i)*H_til;
    [~, I_p1s3_a_til]= PCG(P_p0s3_til, I_H_til, S_til, gamma_til, zeros(N*nx,1), 1e-8, pcg_max_iter);
    
    disp(['alpha = ',num2str(alpha(i)) ' PCG iterations for P*S p1s3 = ', num2str(I_p1s3_a) ' for P_til*S_til p1s3 = ', num2str(I_p1s3_a_til) ])
end

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

function writeBlkPentaDiagMatrixToFile(D, O_up2, O_down2, N, nx, filename)
    % (D, O_up1, O_down1, O_up2, O_down2) forms a block pengta-diagonal matrix
    % O_up1 and O_down1 are all zeros, so the result appears like a block
    % tridiagonal matrix 
    % each block is of size nx
    % size of D = N
    % size of O_up2, O_down2 = N-2
    out = zeros(N*3, nx*nx);
    %  out(1, :) = 0
    out(2, :) = D{1}(:);
    out(3, :) = O_up2{1}(:);
    %  out(4, :) = 0
    out(5, :) = D{2}(:);
    out(6, :) = O_up2{2}(:);
    for i=3:N-2
        offset = (i-1)*3;
        out(offset+1, :) = O_down2{i-2}(:);
        out(offset+2, :) = D{i}(:);
        out(offset+3, :) = O_up2{i}(:);
    end
    out(end-5, :) = O_down2{N-3}(:);
    out(end-4, :) = D{N-1}(:);
    %  out(end-3, :) = 0
    out(end-2, :) = O_down2{N-2}(:);
    out(end-1, :) = D{N}(:);
    %  out(end, :) = 0
    writematrix(out, filename);
end

function writeBlkTriDiagSymMatrixToFile(D, O, N, nx, filename)
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
    out(end-2, :) = tmp(:);
    out(end-1, :) = D{N}(:);
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

function [D_H, O_up2, O_down2, I_H, P] = formPolyPreconditionerHRaw(D, O, N, nx, alpha)
    % this is for split = 3
    % split = 3 -> left & right stair splitting + diagonal splitting
    
    O_up1 = cell(1, N-1);
    O_down1 = cell(1, N-1);
    for i=1:N-1
        O_up1{i} = zeros(nx);
        O_down1{i} = zeros(nx);
    end
    O_up2 = cell(1, N-2);
    O_down2 = cell(1, N-2);
    D_H = cell(1, N);
    D_H{1} = alpha*(inv(D{1}) * O{1} * inv(D{2}) * O{1}') + eye(nx);
    D_H{N} = alpha*(inv(D{N}) * O{N-1}' * inv(D{N-1}) * O{N-1}) + eye(nx);
    for i=2:N-1
        D_H{i} = alpha*(inv(D{i}) * O{i} * inv(D{i+1}) * O{i}' + inv(D{i}) * O{i-1}' * inv(D{i-1}) * O{i-1}) + eye(nx);
        O_up2{i-1} = alpha*(inv(D{i-1}) * O{i-1} * inv(D{i}) * O{i});
        O_down2{i-1} = alpha*(inv(D{i+1}) * O{i}' * inv(D{i}) * O{i-1}');
    end
    I_H = composeBlockPentDiagMatrix(D_H, O_up1, O_up2, O_down1, O_down2, N, nx);
    [~, ~, add] = formPreconditionerSS(D, O, N, nx);
    P = I_H * add;
end

function [D_H, O_up2, O_down2, H] = formPolyPreconditionerH(D, O, N, nx)
    % this is for split = 3
    % split = 3 -> left & right stair splitting + diagonal splitting
    [~, O_add, ~] = formPreconditionerSS(D, O, N, nx);
    
    O_up1 = cell(1, N-1);
    O_down1 = cell(1, N-1);
    for i=1:N-1
        O_up1{i} = zeros(nx);
        O_down1{i} = zeros(nx);
    end
    O_up2 = cell(1, N-2);
    O_down2 = cell(1, N-2);
    D_H = cell(1, N);
    D_H{1} = -O_add{1} * O{1}';
    D_H{N} = -O_add{N-1}'* O{N-1};
    for i=2:N-1
        D_H{i} = -O_add{i} * O{i}' - O_add{i-1}'* O{i-1};
        O_up2{i-1} = -O_add{i-1} * O{i};
        O_down2{i-1} = -O_add{i}'* O{i-1}';
    end
    H = composeBlockPentDiagMatrix(D_H, O_up1, O_up2, O_down1, O_down2, N, nx);
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

function out = composeBlockPentDiagMatrix(D, O_up1, O_up2, O_down1, O_down2, N, nx)
    % need N >= 4
    % D length = N
    % O_up1, O_down1 = N - 1
    % O_up2, O_down2 = N - 2
    out = zeros(N*nx);
    out(1:nx, 1:3*nx) = [D{1}, O_up1{1}, O_up2{1}];
    out(1+nx:2*nx, 1:4*nx) = [O_down1{1}, D{2}, O_up1{2}, O_up2{2}];
    for i=3:N-2
        out((i-1)*nx+1:i*nx, (i-3)*nx+1:(i+2)*nx) = [O_down2{i-2}, O_down1{i-1}, D{i}, O_up1{i}, O_up2{i}];
    end
    out((N-2)*nx+1:(N-1)*nx, (N-4)*nx+1:end) = [O_down2{N-3}, O_down1{N-2}, D{N-1}, O_up1{N-1}];
    out((N-1)*nx+1:end, (N-3)*nx+1:end) = [O_down2{N-2}, O_down1{N-1}, D{N}];
end

function [lambda, i] = PCG(Pinv1, Pinv2, S, gamma, lambda_0, tol, max_iter)
    lambda = lambda_0;
    r = gamma - S*lambda;
    p = Pinv1*r;
    p = Pinv2*p;
    r_tilde = p;
    nu = r'*r_tilde;
    for i=1:max_iter
       tmp = S*p;
       alpha = nu / (p'*tmp);
       r = r - alpha*tmp;
       lambda = lambda + alpha*p;
       r_tilde = Pinv1*r;
       r_tilde = Pinv2*r_tilde;
       nu_prime = r'*r_tilde;
       if abs(nu_prime)<tol
          break
       end
       beta = nu_prime / nu;
       p = r_tilde + beta*p;
       nu = nu_prime;
    end
end