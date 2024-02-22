n1 = 64; n2 = 10007; m1 = 1651; p1  = 7868; 
alpha_initial = normrnd(0, 0.01, [m1, 1]);
old = [0.01,0.01,0.01, alpha_initial'];
%lambda = 100; a_para=0;
%k = [repmat(5, 1, 5), repmat(10, 1, 10), repmat(6, 1, 6), repmat(11, 1, 11), ...
%         repmat(6, 1, 6), repmat(5, 1, 10), repmat(9, 1, 9), repmat(6, 1, 6), ...
%         repmat(14, 1, 14), repmat(8, 1, 8), repmat(6, 1, 6), repmat(8, 1, 8), ...
%         repmat(7, 1, 14), repmat(11, 1, 11), repmat(5, 1, 5), repmat(7, 1, 7), ...
%         repmat(18, 1, 18), repmat(5, 1, 5), repmat(9, 1, 9), repmat(0, 1, 683)];

 k = [repmat(6, 1, 6), repmat(8, 1, 8), repmat(13, 1, 13), repmat(5, 1, 5), ...
         repmat(6, 1, 6), repmat(8, 1, 8), repmat(6, 1, 6), repmat(5, 1, 35), ...
         repmat(10, 1, 10), repmat(6, 1, 6), repmat(5, 1, 5), repmat(6, 1, 6), ...
         repmat(1, 1, 640)];


%%% get initial values
% Convert table to matrix
Y_mat = readtable("y.csv");
wg1_mat = readtable("wg1.csv");
wg1_mat = table2array(wg1_mat);
Y_mat = table2array(Y_mat);
% Perform the matrix operations
u = pinv(wg1_mat' * wg1_mat) * wg1_mat' * Y_mat;

wg2_mat = readtable("wg2_train.csv");
wg2_mat = table2array(wg2_mat);
X = wg2_mat * u;
newz_mat = readtable("z_train.csv", 'ReadVariableNames', false);
newz_mat = table2array(newz_mat);

% Compute the OLS estimate for alpha
alpha1 = pinv(X' * X) * X' * newz_mat; 

% Estimating residuals
e1 = Y_mat - wg1_mat * u;
e2 = newz_mat - X * alpha1;

% Estimating sigmas
sigma1 = std(e1);
sigma2 = std(e2);
sigmau = std(u);
old = [mean(sigma1), mean(sigma2), mean(sigmau), alpha1'];
old(1)=0.01;



%%
n1 = 64; n2 = 10007; m1 = 1651; p1  = 7868; 
element_lengths = [3 7 4 13 4 12 3 4 3 5 3 3 4 3 3 14 4 3 3 6 6 6 4 3 8 4 4 3 8 3 5 3 4 10 3 3 3 3 6 6 4 9 6 3 4 4 5 11 4 4 3 19 4 10 12 6 6 5 3 4 14 5 3 3 3 3 8 8 6 3 3 3 7 7 3 9 4 8 5 3 3 3 11 15 4 6 5 3 6 3 3 3 3 3 3 8 3 3 8 4 5 6 6 3 4 3 8 3 3 6 11 4 3 4 3 3 9 3 3 3 3 6 3 3 3 3 5 6 4 8 21 3 10 4 6 4 5 5 5 3 7 3 4 3 33 3 22 8 15 3 12 4 8 3 3 3 5 3 3 6 3 3 3 4 3 3 8 3 13 3 5 6 4 8 6 3 3 5 4 3 5 4 5 4 3 4 4 3 3 3 4 3 3 5 5 5 3 4 3 5 10 4 4 4 6 5 6 3 4 3 4 4 3 3 5 5 4 3 4 3];
new_vector = repelem(element_lengths, element_lengths);
k_vec = [new_vector, repmat(1, 1, 506)];
% This vector construction assumes a repeating pattern similar to what you've described,
% with adjustments to fit the described sequence. Adjust as necessary for your exact needs.


a = 0;
lam = 75; % Set lambda to 0
tic
% Call the EM_updated function once with lam = 0
disp('Starting EM_updated function for lam = 75');
results = EM_updated(old, a, wg1_mat, wg2_mat, newz_mat, Y_mat, lam, k, m1, n1, n2, p1);
disp('Finished EM_updated function for lam = 75');
toc

% Save the results
save('EM_75_0.mat', 'results');

% lam =75 is the best model

%%
Y_mat = readtable("y.csv");
wg1_mat = readtable("wg1.csv");
wg1_mat = table2array(wg1_mat);
Y_mat = table2array(Y_mat);
% Perform the matrix operations
u = pinv(wg1_mat' * wg1_mat) * wg1_mat' * Y_mat;

wg2_mat = readtable("wg2_test.csv");
wg2_mat = table2array(wg2_mat);
X = wg2_mat * u;
newz_mat = readtable("z_test.csv", 'ReadVariableNames', false);
newz_mat = table2array(newz_mat);
% Compute the OLS estimate for alpha
alpha1 = pinv(X' * X) * X' * newz_mat; 

% Estimating residuals
e1 = Y_mat - wg1_mat * u;
e2 = newz_mat - X * alpha1;

% Estimating sigmas
sigma1 = std(e1);
sigma2 = std(e2);
sigmau = std(u);
old = [mean(sigma1), mean(sigma2), mean(sigmau), alpha1'];
old(1)=0.01;

element_lengths = [3 7 4 13 4 12 3 4 3 5 3 3 4 3 3 14 4 3 3 6 6 6 4 3 8 4 4 3 8 3 5 3 4 10 3 3 3 3 6 6 4 9 6 3 4 4 5 11 4 4 3 19 4 10 12 6 6 5 3 4 14 5 3 3 3 3 8 8 6 3 3 3 7 7 3 9 4 8 5 3 3 3 11 15 4 6 5 3 6 3 3 3 3 3 3 8 3 3 8 4 5 6 6 3 4 3 8 3 3 6 11 4 3 4 3 3 9 3 3 3 3 6 3 3 3 3 5 6 4 8 21 3 10 4 6 4 5 5 5 3 7 3 4 3 33 3 22 8 15 3 12 4 8 3 3 3 5 3 3 6 3 3 3 4 3 3 8 3 13 3 5 6 4 8 6 3 3 5 4 3 5 4 5 4 3 4 4 3 3 3 4 3 3 5 5 5 3 4 3 5 10 4 4 4 6 5 6 3 4 3 4 4 3 3 5 5 4 3 4 3];
new_vector = repelem(element_lengths, element_lengths);
k = [new_vector, repmat(1, 1, 506)];
n1 = 64; n2 = 10008; m1 = 1651; p1  = 7868; 
a = 0.5;
lam = 75; % Set lambda to 0

tic
% Call the EM_updated function once with lam = 0
disp('Starting EM_updated function for lam = 75');
results = EM_updated(old, a, wg1_mat, wg2_mat, newz_mat, Y_mat, lam, k, m1, n1, n2, p1);
disp('Finished EM_updated function for lam = 75');
toc

% Save the results
save('EM_75_v2.mat', 'results');


%%% comm
% Loop through each element of y_gene_unique.Description
y_gene_unique = readtable("y_gene_unique.csv");
ytable = table2cell(y_gene_unique); % Convert to cell array
y = ytable'; y(1, :) = [];
y = cell2mat(y);

w1_geno= readtable("w1_geno.csv");
wg1_mat = readtable("wg1.csv");
wg1_mat = table2array(wg1_mat);
wg2_mat = readtable('wg2.csv');
wg2_mat = table2array(wg2_mat);
newz_mat = readtable("z.csv");
newz_mat = table2array(newz_mat);

u = pinv(wg1_mat' * wg1_mat) * wg1_mat' * y;

X = wg2_mat * u;
% Compute the OLS estimate for alpha
alpha1 = pinv(X' * X) * X' * newz_mat; 

% Estimating residuals
e1 = y - wg1_mat * u;
e2 = newz_mat - X * alpha1;

% Estimating sigmas
sigma1 = std(e1);
sigma2 = std(e2);
sigmau = std(u);
old = [mean(sigma1), mean(sigma2), mean(sigmau), alpha1'];
old(1)=0.01;


n1 = 64; n2 = 20015; m1 = 693; p1  = 4923; 
for i = 1:length(y_gene_unique.Description)
    % Find all indices where y_gene.Description{i} matches w1_geno.gene
    matched_indices{i} = find(strcmp(w1_geno.gene, y_gene_unique.Description{i}));
end
mle_comm = zeros(693, 6);
for i = 1:693
    ynew = y(:,i);
    snpindex = matched_indices{i};
    w1new = wg1_mat(:,snpindex); w2new = wg2_mat(:,snpindex);
    znew = newz_mat;
    % Assuming EM_comm is a function defined elsewhere in your MATLAB code
    em1 = EM_comm([0.01, mean(sigma2), mean(sigmau), alpha1(i)], w1new, ynew, w2new, znew, n1, n2, 10, 0.001);
    mle_comm(i, :) = em1;
end
save('mle_comm_rep.mat', 'mle_comm');


%% multivariate IWAS
Y_mat = readtable("y.csv");
wg1_mat = readtable("wg1.csv");
wg1_mat = table2array(wg1_mat);
Y_mat = table2array(Y_mat);

wg2_mat = readtable('wg2.csv');
wg2_mat = table2array(wg2_mat);
newz_mat = readtable("z.csv");
newz_mat = table2array(newz_mat);
n1 = 64; n2 = 20015; m1 = 1651; p1  = 7868; 

% Assuming u_i_est dimensions are corrected to 180 x 900
u_i_est = zeros(m1, p1); % Initialize the matrix with zeros

% Assuming y, wg1, and wg2 are already defined in your MATLAB workspace
for i = 1:m1
    y_i = Y_mat(:, i);
    % More numerically stable OLS estimation
  u_i_est(i, :) = pinv(wg1_mat) * y_i;
end

% Corrected multiplication to get the predictor
predictor = wg2_mat * u_i_est'; 

% Fitting linear model without an intercept
model = fitlm(predictor, newz_mat, 'Intercept', false);

% Extracting coefficients and p-values
mle_multi = model.Coefficients.Estimate;
p_values_multi = model.Coefficients.pValue;
save('mle_multi.mat', 'mle_multi');
%%PrediXcan can be done in R
