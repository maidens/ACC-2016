%% Define the problem to be solved 

% model parameters 
syms beta gamma A0 kTRANS kPL R1P R1L
unknown_parameters             = [kTRANS kPL]
unknown_parameters_nominal_val = [0.055 0.07]; 
known_parameters     = [beta   gamma A0   R1P  R1L] 
known_parameters_val = [3.4658 2     1e04 1/10 1/10]; 
parameters = [unknown_parameters known_parameters]; 
parameters_val = [unknown_parameters_nominal_val known_parameters_val]; 

% time horizon 
N = 30; 

% flip angle sequence 
alpha = 15*pi/180*ones(2, 1); 

% sampling interval 
dt = 2; 

% system matrices 
A_mini = [-R1P-kPL-(1 - cos(alpha(1, 1)))/dt       0; 
               kPL             -R1L-(1 - cos(alpha(2, 1)))/dt]; 
B_mini = [kTRANS; 0]; 

% discretize with zero order hold 
Abar = expm(dt*A_mini); 
Bbar = inv(A_mini)*(Abar - eye(size(A_mini, 1)))*B_mini; 

% full discrete-time system 
Ad = [3*exp(-dt/beta)               -3*exp(-2*dt/beta)     exp(-3*dt/beta)     0           0; 
             1                              0                     0            0           0; 
             0                              1                     0            0           0; 
      A0*exp(-dt/beta)*Bbar(1)  A0*exp(-2*dt/beta)*Bbar(1)        0        Abar(1, 1)  Abar(1, 2);
      A0*exp(-dt/beta)*Bbar(2)  A0*exp(-2*dt/beta)*Bbar(2)        0        Abar(2, 1)  Abar(2, 2)]; 
Bd = [1; 0; 0; 0; 0]; 
C = [0  0  0  sin(alpha(1, 1))  0; 
     0  0  0  0                 sin(alpha(2, 1))]; 

% initial condition 
x0 = zeros(size(Ad, 1), 1); 

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
K = [0 0; 0 1]; 

% define covariance matrix Sigma
Sigma = eye(2); 


%% Translate this problem into a QP with objective function u'*Q*u + q'*u + q0

disp('==== Computing objective function ====') 

[ Q, q, q0 ] = compute_objective_function( Ad_val, Bd_val, C, x0_val, dA_val, dB_val, dx0_val, K, Sigma, N ); 
  

%%  Solve semidefinite relaxation of l2-constrained problem 

disp('==== Solving SDP relaxation of l2-constrained problem ====') 

scale_factor = 1e-9
Q_scaled = scale_factor*Q; 
cvx_begin sdp 
    variable U(N, N) symmetric 
    for i=1:N
        U(i, i) <= 1; 
    end
    U >= 0; 
    trace(U) <= 16; 
    maximize trace(Q_scaled*U) 
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
% title('Optimal infusion input') 
xlabel('time (s)') 
ylabel('infusion strength') 
set(gca,'FontSize',20);
tightfig(gcf); 
print(gcf, '-dpdf', 'l2-optimal_input.pdf');


%% Compute state and output trajectories using this optimized input sequence

% propogate state forward in time 
x(:, 1) = x0; 
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
leg = legend('pyruvate', 'lactate'); 
% title('Observed output using optimal input') 
xlabel('time (s)')
ylabel('observed signal (au)') 
set(leg,'FontSize',20); 
set(gca,'FontSize',20);
tightfig(gcf); 
print(gcf, '-dpdf', 'l2-optimal_outputs.pdf');


%%  Solve semidefinite relaxation of l1-constrained problem 

disp('==== Solving SDP relaxation of l1-constrained problem ====') 

total = 8; 
cvx_begin sdp 
    variable U(N, N) symmetric 
    variable u(N, 1)
    [U u; u' 1] >= 0
    trace(ones(N,N)*U) <= total^2
    maximize trace(Q_scaled*U)
    for i=1:N
        0 <= U(i, i) <= u(i) <= 1
        for j=1:N
             0 <= U(i, j) 
        end
    end
cvx_end


%% Compare the results with a simple boxcar input 

u_star = [ones(total, 1); zeros(N-total, 1)]; 
optimal_value_relaxation = cvx_optval/scale_factor
value_boxcar = u_star'*Q_scaled*u_star/scale_factor
approximation_accuracy = value_boxcar/optimal_value_relaxation


%% Plot the boxcar

figure 
plot(dt*(0:N-1), u_star, 'ko-',  'MarkerFaceColor', berkeley_colors(3, :), 'LineWidth', 2) 
% title('Boxcar infusion input') 
xlabel('time (s)') 
ylabel('infusion strength') 
set(gca,'FontSize',20);
tightfig(gcf); 
print(gcf, '-dpdf', 'boxcar_input.pdf');


%% Compute state and output trajectories using this optimized input sequence

% propogate state forward in time 
x(:, 1) = x0; 
for t=1:N
    x(:, t+1) = Ad_val*x(:, t) + Bd_val*u_star(t); 
end
y = C*x; 

% plot the results 
figure
set(gca,'ColorOrder', berkeley_colors, 'NextPlot', 'replacechildren')
hold on
plot(dt*(0:N), y(1, :),  'o-', 'MarkerFaceColor', berkeley_colors(1, :), 'LineWidth', 2)
plot(dt*(0:N), y(2, :),  'o-', 'MarkerFaceColor', berkeley_colors(2, :), 'LineWidth', 2)
hold off
leg = legend('pyruvate', 'lactate');  
% title('Observed output using boxcar input') 
xlabel('time (s)')
ylabel('observed signal (au)') 
set(leg,'FontSize',20); 
set(gca,'FontSize',20);
tightfig(gcf); 
print(gcf, '-dpdf', 'boxcar_outputs.pdf');



