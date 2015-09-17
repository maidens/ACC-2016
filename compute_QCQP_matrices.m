function [ Q, q, q0 ] = compute_QCQP_matrices( A, B, C, x0, dA, dB, dx0, K, Sigma, N )

warning('off', 'MATLAB:nearlySingularMatrix')

% number of parameters of interest
p = size(dA, 3); 

% compute m 
for t=0:N
    for i=1:p
        m(:, i, t+1) = A^t*dx0(:, i); 
        for l=0:t-1
            m(:, i, t+1) = m(:, i, t+1) + A^(t-l-1)*dA(:, :, i)*A^l*x0; 
        end
    end
end

% compute M 
for t=0:N 
    for k=0:N-1 
        for i=1:p
            M(:, :, i, k+1, t+1) = A^(t-k-1)*dB(:, :, i);
            for l=k+1:t-1
                M(:, :, i, k+1, t+1) = M(:, :, i, k+1, t+1) + A^(t-l-1)*dA(:, :, i)*A^(l-k-1)*B; 
            end
        end
    end
end

% compute S 
S = C'*inv(Sigma)*C; 

% compute q0 
q0 = 0; 
for t=0:N
    for h=1:size(A, 1)
        for hp=1:size(A, 1)
            for i=1:p
                for ip=1:p
                    q0 = q0 + K(i, ip)*m(hp, ip, t+1)*S(h, hp)*m(h, i, t+1); 
                end
            end
        end
    end
end

%  compute q 
nu = size(B, 2); 
q = zeros(N*nu, 1); 
for j=1:nu
    for k=0:N-1
        for t=k+1:N
            for h=1:size(A, 1)
                for hp=1:size(A, 1)
                    for i=1:p
                        for ip=1:p
                            q(k*nu+j, 1) = q(k*nu+j, 1) + K(i, ip)*m(hp, ip, t+1)*S(h, hp)*M(h, j, i, k+1, t+1); 
                        end
                    end
                end
            end
        end
    end
end
        
%  compute Q 
Q = zeros(N*nu, N*nu); 
for j=1:nu
    for jp=1:nu
        for k=0:N-1
            for kp=0:N-1
                for t=max([k kp])+1:N
                    for h=1:size(A, 1)
                        for hp=1:size(A, 1)
                            for i=1:p
                                for ip=1:p
                                    Q(k*nu+j, kp*nu+jp) = Q(k*nu+j, kp*nu+jp) + K(i, ip)*M(hp, jp, ip, kp+1, t+1)*S(h, hp)*M(h, j, i, k+1, t+1); 
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

warning('on', 'MATLAB:nearlySingularMatrix')


end

