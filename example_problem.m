%% Define the problem to be solved 

% model parameters 
syms alpha_0 beta x10 x20 
unknown_parameters             = [beta x10 x20]
unknown_parameters_nominal_val = [1  0.1  0.2]; 
known_parameters     = [alpha_0] 
known_parameters_val = [0.5]; 
parameters = [unknown_parameters known_parameters]; 
parameters_val = [unknown_parameters_nominal_val known_parameters_val]; 

% time horizon 
N = 30; 

% sampling interval 
dt = 0.1; 

% system matrices 
A = [-beta    0;
      beta -alpha_0];     
B = [1; 0]; 
C = [1 0; 
     0 1]; 
D = [0; 0]; 

% discretized dynamics 
disp('==== Computing discretized dynamics ====') 
Ad = expm(dt*A); 
Bd = inv(A)*(Ad - eye(size(A, 1)))*B; 

% initial condition 
x0 = [x10; x20]; 

% evaluate nominal values of system matrices and initial state 
x0_val = double(subs(x0, parameters, parameters_val)); 
Ad_val = double(subs(Ad, parameters, parameters_val)); 
Bd_val = double(subs(Bd, parameters, parameters_val));

% compute nominal parametric sensitivities 
disp('==== Computing sensitivities ====') 
for i=1:length(unknown_parameters)
    dx0_val(:, i) = double(subs(diff(x0, unknown_parameters(i)), parameters, parameters_val)); 
    dA_val(:, :, i)  = double(subs(diff(Ad, unknown_parameters(i)), parameters, parameters_val)); 
    dB_val(:, :, i)  = double(subs(diff(Bd, unknown_parameters(i)), parameters, parameters_val)); 
end

% define linear objective function matrix K 
K = eye(3); 

% define covariance matrix Sigma
Sigma = eye(1); 


%% Translate this problem into a QP with objective function u'*Q*u + q'*u + q0

disp('==== Computing QCQP matrices ====') 

[ Q, q, q0 ] = compute_QCQP_matrices( Ad_val, Bd_val, C, x0_val, dA_val, dB_val, dx0_val, K, Sigma, N ); 
  

%%  Solve semidefinite relaxation of l2-constrained problem 

disp('==== Solving SDP relaxation of l2-constrained problem ====') 

scale_factor = 1
Q_scaled = scale_factor*Q; 
q_scaled = scale_factor*q; 
q0_scaled = scale_factor*q0
obj_matrix = [Q_scaled q_scaled; q_scaled'  q0];     

cvx_begin sdp 
 %   variable V(N+1, N+1) symmetric
    variable U(N, N) symmetric 
    variable u(N, 1)
    for i=1:N
        0<= U(i, i) <= 1
    end
    U_lifted = [U u; u' 1]
    trace(U) <= 4.5^2
    U_lifted >= 0
    maximize trace(obj_matrix*U_lifted) 
cvx_end


%% Extract solution to QP from semidefinite relaxation solution

% ensure that U has rank 1 
epsilon = 1e-06; 
assert(rank(U, epsilon) == 1) 

% compute decomposition U = u_opt*u_opt' 
[V, D] = eig(U); 
u_opt = abs(V(:, N)*sqrt(D(N, N))); 

% plot the results

berkeley_colors = ...
 1/256*[ 45,  99, 127; 
        224, 158,  25; 
          0,   0,   0;
        194, 185, 167;
        217, 102, 31;
        185, 211, 182]; 
    
    
%%    
figure 
plot(dt*(0:N-1), u_opt, 'ko-',  'MarkerFaceColor', berkeley_colors(3, :), 'LineWidth', 2) 
axis([0 N*dt 0 1.1])
title('Optimal input') 
xlabel('t') 
ylabel('u(t)') 
tightfig(gcf); 
print(gcf, '-dpdf', 'toy_problem_input.pdf');


%% Compute state and output trajectories using this optimized input sequence

% propogate state forward in time 
x(:, 1) = x0_val; 
for t=1:N
    x(:, t+1) = Ad_val*x(:, t) + Bd_val*u_opt(t); 
end
y = C*x; 

% plot the results 
figure
set(gca,'ColorOrder', berkeley_colors, 'NextPlot', 'replacechildren')
hold on
plot(dt*(0:N), y(1, :),  'o-', 'MarkerFaceColor', berkeley_colors(1, :), 'LineWidth', 2)
plot(dt*(0:N), y(2, :),  'o-', 'MarkerFaceColor', berkeley_colors(2, :), 'LineWidth', 2)
hold off
legend('y_1', 'y_2') 
title('Output from nominal system using optimal input') 
axis([0 3 0 1.4])
xlabel('t')
ylabel('y(t)') 
tightfig(gcf); 
print(gcf, '-dpdf', 'toy_problem_outputs.pdf');

