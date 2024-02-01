function [output] = mstep_updated(old, a_para, wg1, wg2, z, y, lambda, k, m1, n1, n2, p1)
sigma1 = old(1);
sigma2 = old(2);
sigmau = old(3);
alpha_g = old(4:end);
%a = a_para;

% Compute common matrices and vectors
tStart = tic;
inverse_common = (1 / sigma1) * (wg1' * wg1);
%wg2_transposed = wg2';
%wg1_transposed = wg1';
%z_transposed = z';


% Preallocate results for better performance
results = repmat(struct('E1', [], 'E2', [], 'E3', [], 'alphaest', []), 1, m1);


% Initialize accumulation variables
% Note: Accumulation variables inside parfor need special handling
%total_u_estimation_times = zeros(1, m1);
%total_alpha_estimation_times = zeros(1, m1);
%total_like_estimation_times = zeros(1, m1);

%[total_u_estimation_times,total_alpha_estimation_times,total_like_estimation_times] = deal(zeros(1, m1),zeros(1, m1),zeros(1, m1));

parfor i = 1:m1
    disp(['Index: ' num2str(i) ' in m1 using mstep_updated function.']);

    % Iteration-specific calculations
    %current_alpha_g = alpha_g(i);
    A = inverse_common + (alpha_g(i)^2) / sigma2 * wg2' * wg2 + 1 / sigmau * eye(p1);
    A = A + 1e-4 * eye(size(A)); % Small regularization term
    I = eye(size(A)); % Create an identity matrix of the same size as A
    sigma_ui_i = A \ I; % Compute the inverse
    mu_ui_i = sigma_ui_i * (1 / sigma1 * wg1' * y(:, i) + alpha_g(i) / sigma2 * wg2' * z);

    si = 2 * z' * wg2 * mu_ui_i - a_para * lambda * sign(alpha_g(i));
    abc_i = (1 - a_para) * lambda * k(i);
    if abs(si) <= abc_i || abs(si) <= (a_para * lambda)
        alphaest = 0;
    else
        alphaest = (1 - abc_i / abs(si)) * (2 * mu_ui_i' * wg2' * wg2 * mu_ui_i + 2 * trace(wg2 * sigma_ui_i * wg2'))^(-1) * si;
    end

    E1 = y(:, i)' * y(:, i) - 2 * y(:, i)' * wg1 * mu_ui_i + mu_ui_i' * wg1' * wg1 * mu_ui_i + trace(wg1' * wg1 * sigma_ui_i);
    E2 = z' * z - 2 * alphaest * z' * wg2 * mu_ui_i + (alphaest * wg2 * mu_ui_i)' * (alphaest * wg2 * mu_ui_i) + trace(alphaest^2 * wg2 * sigma_ui_i * wg2');
    E3 = mu_ui_i' * mu_ui_i + trace(sigma_ui_i);
    % Store results for this iteration
    results(i) = struct('E1', E1, 'E2', E2, 'E3', E3, 'alphaest', alphaest);
    disp(['Finished index : ' num2str(i) ' in m1 using mstep_updated function.']);
end

% Compute final estimates
sigma1est = 1 / (m1 * n1) * sum([results.E1]);
sigma2est = 1 / (m1 * n2) * sum([results.E2]);
sigmauest = 1 / (m1 * p1) * sum([results.E3]);
final_estimation_time = toc(tStart);
fprintf('Final Estimation Calculation Time: %d minutes and %f seconds\n', floor(final_estimation_time/60), rem(final_estimation_time,60));
 %disp(['Final Estimation Calculation Time: ', num2str(final_estimation_time), ' seconds']);

log_likelihood_y = -0.5 * n1 * m1 * log(2 * pi * sigma1est) - 0.5 * sum([results.E1]) / sigma1est;
log_likelihood_z = -0.5 * n2 * log(2 * pi * sigma2est) - 0.5 * sum([results.E2]) / sigma2est;
log_likelihood_u = -0.5 * p1 * m1 * log(2 * pi * sigmauest) - 0.5 * sum([results.E3]) / sigmauest;
log_likelihood = log_likelihood_y + log_likelihood_z + log_likelihood_u;
all_E2_values = arrayfun(@(x) x.E2, results);
E2_null = z' * z
 output = [sigma1est, sigma2est, sigmauest, [results.alphaest],all_E2_values ,E2_null,log_likelihood];
end
