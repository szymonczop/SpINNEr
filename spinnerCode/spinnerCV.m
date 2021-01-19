%-------------------------------------------------------------------------- 
% This function finds the optimal pair of regularization parameters, 
% lambda_N and lambda_L, and finds SpINNEr estimates by solving the
% problem:
%--------------------------------------------------------------------------
%    argmin_{B, beta} {   0.5*sum_i ( y_i - X*beta - <A_i, B> )^2 + 
%                                 lambda_N*|| B ||_* + 
%                               lambda_L*|| vec(B o W) ||_1    }
%--------------------------------------------------------------------------
% In order to do so, two-dimensional cross-validation is applied and the 
% specific implementation of ADMM algorithm is used

% The SpINNEr toolbox is free software: you can redistribute it and/or 
% modify it under the terms of the GNU General Public License as published 
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% The SpINNEr toolbox is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Authors:    Damian Brzyski and Xixi Hu
%         Date:       June 27, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%-------------------------------------------
% REQUIRED INPUTS ARGUMENTS:               -
%-------------------------------------------
% y:                 The n-dimensional vector of responses
%~~~~~~~~~~~~~~~~~~~~~
% X:                 matrix of covariates included in the model but not
%                    penalized. Symbol "[]" should be provided to omit it.
%~~~~~~~~~~~~~~~~~~~~~
% AA:                This should be three-dimensional array with dimensions
%                    p,p and n, where ith slice, i.e. A(:,:,i), is a 
%                    symmetric matrix with zeros on the diagonal.
%                    The alternative form, AA = [A1, A2, A3,...,An],is also
%                    supported
%
%-------------------------------------------
% OPTIONAL INPUT ARGUMENTS (provided in standard name-value pair fashion)
%-------------------------------------------
% 'UseParallel':    Turns on/off the parallel computation for 
%                   cross-validation. Possible falues are: false (default)
%                   and true
%~~~~~~~~~~~~~~~~~~~~~
% 'W':              Matrix of weights in the SpINNEr optimization problem.
%                   Symmetric p-by-p matrices with nonnegative entries are
%                   possible. The default W has zeros on the diagonal and
%                   ones on the off-diagonal entries
%~~~~~~~~~~~~~~~~~~~~~
%'gridLengthN'      The number of different values of lambda_N considered
%                   in the cross-validation. These values are automatically
%                   selected.
%~~~~~~~~~~~~~~~~~~~~~
%'gridLengthL'      The number of different values of lambda_L considered
%                   in the cross-validation. These values are automatically
%                   selected.   
%~~~~~~~~~~~~~~~~~~~~~
%'kfolds'           The number of folds considered in cross-validation
%~~~~~~~~~~~~~~~~~~~~~
%'displayStatus'    Turns on/off the messages generated after checking each
%                   pair of considered in the cross-validation parameters. 
%                   Possible falues are: false and true (default).
%~~~~~~~~~~~~~~~~~~~~~

%%
%-------------------------------------------
%              OUTPUTS:                    -
%-------------------------------------------
% Outputs can be obtained from the structure class object, 'out'.
%~~~~~~~~~~~~~~~~~~~~~
% B                 SpINNEr estimate of B
%~~~~~~~~~~~~~~~~~~~~~
% beta              SpINNEr estimate of beta
%~~~~~~~~~~~~~~~~~~~~~
% bestLambdaN       The best lambda_N found by cross-validation
%~~~~~~~~~~~~~~~~~~~~~
% bestLambdaL       The best lambda_L found by cross-validation
%~~~~~~~~~~~~~~~~~~~~~
% LambsNgrid        Values of lambda_N considered in cross-validation
%~~~~~~~~~~~~~~~~~~~~~
% LambsLgrid        Values of lambda_L considered in cross-validation
%~~~~~~~~~~~~~~~~~~~~~
% logliksCV         The matrix of prediction errors estimated by
%                   cross-validation for each considered pair of 
%                   tuning parameters (lambda_Ns are in rows)
%-------------------------------------------

function out = spinnerCV(y, X, AA, varargin)
%% Tensor of regressor matrices
[p1, p2, p3] = size(AA);
if p3==1
    if mod(p2,p1)~=0
        error('The assumed form is AA = [A1, A2,...,An], with square matrices Ai, hence number of columns in AA should be the multiplicity of the number of rows')
    else
        AA = reshape(AA, [p1, p1, p2/p1]);
    end
end
%% Objects
n = length(y);

%% Additional parameters (AP)
AP                   = inputParser;
defaultUseParallel   = false;
defaultW             = ones(p1, p1) - eye(p1, p1);
defaultGridLengthN   = 15;
defaultGridLengthL   = 15;
defaultKfolds        = 5;
defaultDisplayStatus = true;
%------------------------------------
addRequired(AP, 'y');
addRequired(AP, 'X');
addRequired(AP, 'AA');
addOptional(AP, 'UseParallel', defaultUseParallel, @islogical );
addOptional(AP, 'W', defaultW, @(x) and(issymmetric(x), all(all(x>=0))));
addOptional(AP, 'gridLengthN', defaultGridLengthN, @(x) and(x>0, x == round(x)));
addOptional(AP, 'gridLengthL', defaultGridLengthL, @(x) and(x>0, x == round(x)));
addOptional(AP, 'kfolds', defaultKfolds, @(x) and(x>0, x == round(x)));
addOptional(AP, 'displayStatus', defaultDisplayStatus, @islogical );
%------------------------------------
parse(AP, y, X, AA, varargin{:})
%------------------------------------
UseParallel   = AP.Results.UseParallel;
displayStatus = AP.Results.displayStatus;
W             = AP.Results.W;
gridLengthN   = AP.Results.gridLengthN;
gridLengthL   = AP.Results.gridLengthL;
kfolds        = AP.Results.kfolds;

%% CV options
initLambda       = 1;    % first considered lambdaL
zeroSearchRatio  = 100;  % decides on the speed of increasing lambdas (for finding the zero estimate)
maxLambAcc       = 1e-2; % precision for finding regularization parameters which give zero estimate for the first time

%% Cross-validation indices
minElemsN        = floor(n/kfolds);
remainingElemsN  = n-minElemsN*kfolds;
oneToKfols       = 1:kfolds;
GroupsIdxs       = bsxfun(@times,oneToKfols, ones(minElemsN,1));
GroupsIdxs       = [ GroupsIdxs; [oneToKfols(1:remainingElemsN), zeros(1, kfolds-remainingElemsN)] ]; % if n is not a multiplicity of kfolds
GroupsIdxs       = GroupsIdxs(:);
GroupsIdxs(GroupsIdxs == 0) = [];
s                = rng;
rng('default')  % to get the same results every time
randSmple        = randsample(n,n);
GroupsIdxs       = GroupsIdxs(randSmple);
rng(s);

%% Finding the maximal lambda L
clambdaL  = initLambda;
ValsLambL = zeros(1,1);
counterr1 = 1;
stopp     = 0;

% finding lambda_L for which matrix of zeros is obtained
while stopp == 0
    out = spinner(y, X, AA, 0, clambdaL, W);
%    if norm(out.B, 'fro') < 1e-16
    if sqrt(sum(sum(W.*out.B.^2))) < 1e-16
        stopp = 1;
    end
    ValsLambL(counterr1) = clambdaL;
    clambdaL             = zeroSearchRatio*clambdaL;
    counterr1            = counterr1 + 1;
end

% initial interval for maximal lambda L
if length(ValsLambL) == 1
    lamL1 = 0;
    lamL2 = ValsLambL;
else
    lamL1 = ValsLambL(end-1);
    lamL2 = ValsLambL(end);
end

% finding narrow interval for maximal lambda L
stopp      = 0;
counterr2  = 1;
ValsLambLmax = zeros(1,1);
while stopp == 0
    cLamLmaxNew0  = (lamL1 + lamL2)/2;
    outNew0       = spinner(y, X, AA, 0, cLamLmaxNew0, W);
    if norm(outNew0.B, 'fro') < 1e-16
        lamL2 = cLamLmaxNew0;
    else
        lamL1 = cLamLmaxNew0;
    end
    ValsLambLmax(counterr2) = lamL2;
    counterr2 = counterr2 + 1;
    if abs(lamL2 - lamL1)/lamL2 < maxLambAcc
        stopp = 1;
    end
end

%% Finding the maximal lambda N
clambdaN  = initLambda;
ValsLambN = zeros(1,1);
counterr1 = 1;
stopp     = 0;

% finding lambda_N for which matrix of zeros is obtained
while stopp == 0
    out = spinner(y, X, AA, clambdaN, 0, W);
    if norm(out.B, 'fro') < 1e-16
        stopp = 1;
    end
    ValsLambN(counterr1) = clambdaN;
    clambdaN             = zeroSearchRatio*clambdaN;
    counterr1            = counterr1 + 1;
end

% initial interval for maximal lambda N
if length(ValsLambN) == 1
    lamN1 = 0;
    lamN2 = ValsLambN;
else
    lamN1 = ValsLambN(end-1);
    lamN2 = ValsLambN(end);
end

% finding narrow interval for maximal lambda N
stopp      = 0;
counterr2  = 1;
ValsLambNmax = zeros(1,1);
while stopp == 0
    cLamLmaxNew0  = (lamN1 + lamN2)/2;
    outNew0       = spinner(y, X, AA, cLamLmaxNew0, 0, W);
    if norm(outNew0.B, 'fro') < 1e-16
        lamN2 = cLamLmaxNew0;
    else
        lamN1 = cLamLmaxNew0;
    end
    ValsLambNmax(counterr2) = lamN2;
    counterr2 = counterr2 + 1;
    if abs(lamN2 - lamN1)/lamN2 < maxLambAcc
        stopp = 1;
    end
end

%% Final lambdas grids
k           = 0.75;
seqq        = (1:(gridLengthL-1))/(gridLengthL-1);
LambsLgrid  = [0, exp(  ( seqq * log(lamL2+1)^(1/k) ).^k  ) - 1 ];
LambsNgrid  = [0, exp(  ( seqq * log(lamN2+1)^(1/k) ).^k  ) - 1 ];

%% Cross - Validation
logliksCV = zeros(gridLengthN, gridLengthL);
if UseParallel
    parfor ii = 1:gridLengthN
    clambdaN = LambsNgrid(ii);
        for jj = 1:gridLengthL
            clambdaL  = LambsLgrid(jj);
            normResCV = zeros(kfolds,1);
            for gg = 1:kfolds
                testIndices     =  find(GroupsIdxs == gg);
                treningIndices  =  setdiff(1:n , testIndices)';
                AA_trening      =  AA(:,:, treningIndices); %#ok<*PFBNS>
                AA_test         =  AA(:,:, testIndices);
                if ~isempty(X)
                    X_trening       =  X(treningIndices,:);
                    X_test          =  X(testIndices,:);
                else
                    X_trening       =  [];
                    X_test          =  0;
                end
                y_trening       =  y(treningIndices);
                y_test          =  y(testIndices);
                out_CV          =  spinner(y_trening, X_trening, AA_trening, clambdaN, clambdaL, W);
                AA_test_p       =  permute(AA_test, [3 1 2]);
                normResCV(gg)   =  0.5*norm(y_test - AA_test_p(:,:)*out_CV.B(:) - X_test*out_CV.beta)^2;
            end        
            logliksCV(ii, jj) =  sum(normResCV)/n;
            if displayStatus
                disp(strcat(['finished:  ', num2str(ii), endText(ii),' ', 'value from lambdaN grid,  ', num2str(jj), endText(jj), ' value from lambdaL grid']))
            end
        end
    end
else
    for ii = 1:gridLengthN
        clambdaN = LambsNgrid(ii);
        for jj = 1:gridLengthL
            clambdaL  = LambsLgrid(jj);
            normResCV = zeros(kfolds,1);
            for gg = 1:kfolds
                testIndices     =  find(GroupsIdxs == gg);
                treningIndices  =  setdiff(1:n , testIndices)';
                AA_trening      =  AA(:,:, treningIndices); %#ok<*PFBNS>
                AA_test         =  AA(:,:, testIndices);
                if ~isempty(X)
                    X_trening       =  X(treningIndices,:);
                    X_test          =  X(testIndices,:);
                else
                    X_trening       =  [];
                    X_test          =  0;
                end
                y_trening       =  y(treningIndices);
                y_test          =  y(testIndices);
                out_CV          =  spinner(y_trening, X_trening, AA_trening, clambdaN, clambdaL, W);
                AA_test_p       =  permute(AA_test, [3 1 2]);
                normResCV(gg)   =  0.5*norm(y_test - AA_test_p(:,:)*out_CV.B(:) - X_test*out_CV.beta)^2;
            end        
            logliksCV(ii, jj) =  sum(normResCV)/n;
            if displayStatus
                disp(strcat(['finished:  ', num2str(ii), endText(ii),' ', 'value from lambdaN grid,  ', num2str(jj), endText(jj), ' value from lambdaL grid']))
            end
        end
    end
end

%% Optimal lambdas
[MM, II]    = min(logliksCV);
Xindex      = find(MM == min(MM), 1);
Yindex      = II(Xindex);
bestLambdaN = LambsNgrid(Yindex);
bestLambdaL = LambsLgrid(Xindex);

%% Final estimate
outFinal    =  spinner(y, X, AA, bestLambdaN, bestLambdaL, W);

%% Output
out              = struct;
out.LambsLgrid   = LambsLgrid;
out.LambsNgrid   = LambsNgrid;
out.logliksCV    = logliksCV;
out.B            = outFinal.B;
out.beta         = outFinal.beta;
out.bestLambdaN  = bestLambdaN;
out.bestLambdaL  = bestLambdaL;

end