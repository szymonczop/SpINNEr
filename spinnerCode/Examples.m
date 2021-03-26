%% Addpath
addpath('.\spinnerCode') % the path to the directory "spinnerCode" should be placed here

%% Settings
p        = 40;
n        = 100;
sigStr   = 2;

%% Data
% true signal 
rng(1)
B1      = sigStr*ones(5,5);
B2      = -sigStr*ones(5,5);
B3      = sigStr*ones(4,4);
sNods   = size(B1,1) + size(B2,1) + size(B3,1);
B       = blkdiag(zeros(5,5), B1, zeros(6,6), B2, zeros(7,7), B3, zeros(p - sNods - 18, p- sNods - 18));
spinnerHeatmap(B , 'True signal')

% regressor matrices
rng('default')
rng(2021)
AA               = randn([p, p, n]);
AA               = AA + permute(AA, [2 1 3]); % matrices are symmetric
AA2              = reshape(AA, [p^2, n]);
AA2              = zscore(AA2')';             % standardize data
idxsDiag         = logical(reshape(eye(p,p),[p^2,1]));
AA2(idxsDiag,:)  = 0;                         % zeros on the diagonals
AA               = reshape(AA2, [p, p, n]);   % 3-way tensor of symmetric matrices with zeros on diagonals and standardized across 3rd dimension

% response generating
sigma            = 1;               % noise level
eps              = sigma*randn(n,1);
y                = AA2'*B(:) + eps;  % we have:   y_i = <A_i, B> + eps_i

%---------------------------------
%---------------------------------
%%   ESTIMATION   WITH   SPINNER
%---------------------------------
%% The basic settings
% Here we just run spinnerCV with the default settings. Notice that
% setting X = ones(n,1) implies that intercept is included in the model.
spinnerFit     = spinnerCV(y, ones(n,1), AA);
spinnerHeatmap(spinnerFit.B, 'Estimated matrix')

%% Speeding up the cross-validation
% Cross-validation may take longer time so if the parallel computing is an
% option, the easy way to speed the process up is to use spinnerCV with
% an option allowing for parallelization:
spinnerFit2 = spinnerCV(y, ones(n,1), AA, 'UseParallel', true);
spinnerHeatmap(spinnerFit2.B, 'Estimated matrix')

% The messages informing about the progress may be turned off:
spinnerFit3 = spinnerCV(y, ones(n,1), AA, 'UseParallel', true, 'displayStatus', false);

% By default, spinnerCV considers 15 different values for lambdaN and 15
% different values for lambdaL. This can be changed by using the options
% 'gridLengthL' and 'gridLengthN'. To perform cross-validation with only 5
% values for each lambda, it is enough to type:
spinnerFit4 = spinnerCV(y, ones(n,1), AA, 'UseParallel', true, 'gridLengthN', 5, 'gridLengthL', 5);

% The optimial tuning parameters selected by cross-validation, optimalLambdaL 
% and optimalLambdaN can be reached as
optimalLambdaL = spinnerFit3.bestLambdaL;
optimalLambdaN = spinnerFit3.bestLambdaN;

%% Getting the spinner solution for user-defined pair of parameters
% To get the spinner estimate for user-defined pair of tuning parameters,
% one can use the function "spinner". For the lambdas previously selected
% as optimal. This should be done in a following way:
spinnerGivenLambdas = spinner(y, ones(n,1), AA, optimalLambdaN, optimalLambdaL);

% We can easily check that that gives the same estimate as previously
% obtained by spinnerCV:
norm(spinnerFit3.B - spinnerGivenLambdas.B)

%% Using different weights via matrix W
% By default, W is set as the matrix with zeros on the diagonal and ones
% outside its diagonal. SpINNEr can handle any symmetric matrix of nonzero
% entries. For example, one may define W based on the obtained estimate and
% fit the model again, such as the entries already estimated as relatively
% large will not be penalized so much as other. One natural way is to
% define W as:
spEstim   =  spinnerFit2.B;
W         =  max(max(abs(spEstim))) - abs(spEstim);
W         =  W/max(max(abs(W)));
W         =  (W + W')/2;
W         =  W - diag(diag(W));
spinnerHeatmap(W, 'Updated matrix of weights')

% Let us fit the model again with the updated matrix of weights. This can
% be done by using the option 'W', followed by a matrix which one wants to
% use:
spinnWCV  = spinnerCV(y, ones(n,1), AA, 'UseParallel', true, 'W', W);
spinnerHeatmap(spinnWCV.B, 'SpINNEr estimate after updating weights')

%% Finding the optimal order of nodes
% In practice we do not know which nodes can be clustered together and
% therefore we can only assume the structured form of the true signal for
% some (unknown) order of nodes. SpINNEr can also be used to recover such 
% clustering. To see that, suppose that B (defined in line 17) is the true
% signal in the cluster-by-cluster order. Suppose that the permutation:
P = randperm(p);
% restores the alphabetical order of nodes labels. The corresponding
% matrices A_i have their rows and columns permuted by P: 
for ii = 1:n
    AAp(:,:,ii) = AA(P, P, ii); %#ok<SAGROW>
end
% and AAp(:,:,ii) are the starting data for us, i.e. we do not know the
% permutation P at the beginning. We fit SpINNEr:
spFitAlph   = spinnerCV(y, ones(n,1), AAp, 'UseParallel', true);
spEstimAlph = spFitAlph.B;
spinnerHeatmap(spEstimAlph, 'SpINNEr for the initial (e.g. alphabetical) order of nodes')
% there is some structure but it is challenging to see any blocks. Now,
% find the singular value decomposition (SVD) of the estimate:
[U, S, V] = svd(spEstimAlph);
% It means that we have    spFitPerm.B = U*S*V'
% Now, the clusters may be found by looking at the largest magnitudes of a
% few first singular vectors (SV):
plot(U(:,1:3))
legend('first SV','second SV', 'third SV')
% the largest magnitudes can be selected in a various way, for instance the
% methods finding outliers may be useful:
ind1 = find(isoutlier( U(:,1), 'gesd'));
ind2 = find(isoutlier( U(:,2), 'gesd'));
ind3 = find(isoutlier( U(:,3), 'gesd'));
% in the case where there is no overlapping, we can now simply build the
% permutation returning the cluster-by-cluster order as
spinnerP    = [ind1; ind2; ind3; setdiff(1:p, [ind1; ind2; ind3])'];
spEstimPerm = spEstimAlph(spinnerP, spinnerP);
spinnerHeatmap(spEstimPerm, 'The cluster-by-cluster order recovered')





