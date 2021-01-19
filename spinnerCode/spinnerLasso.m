function out = spinnerLasso(y, SVDAx, lambdaL, WGTs, solOptions)

% this function solves the problem 
%
% argmin_B {  0.5*sum_i (y_i - <A_i, B>)^2 + lambda_L || B ||_1  }
% 
% y and AA are after regressing X out

%% Objects
p0           = size(SVDAx.U,1);
p            = (1+sqrt(1+8*p0))/2;

%% Solver options
deltaInitial  = solOptions.deltaInitial2;
mu            = solOptions.mu;
deltaInc      = solOptions.deltaInc;
deltaDecr     = solOptions.deltaDecr;
maxIters      = solOptions.maxIters;
epsPri        = solOptions.epsPri;
epsDual       = solOptions.epsDual;

%% SVD 
U      = SVDAx.U;
Sdiag  = SVDAx.Sdiag;
Vt     = SVDAx.Vt;
idxs   = SVDAx.idxs;

%% Initial primal and dual matrix
Dk = zeros(p,p);
Wk = zeros(p,p);

%% ADMM loop
delta    = deltaInitial;
counterr = 0;
stop     = 0;
while stop == 0
    Bnew = ProxFsvd(y, SVDAx, Dk, Wk, delta);
    Dnew = ProxH_lasso(Bnew, delta, Wk, lambdaL, WGTs);
    Wk   = Wk + delta*(Dnew - Bnew);  
    rk               = Dnew - Bnew;
    sk               = Dnew - Dk;
    rknorm           = norm(rk,'fro');
    Bnorm            = norm(Bnew,'fro');
    rknormR          = rknorm/Bnorm;
    sknorm           = norm(sk,'fro');
    sknormR          = sknorm/norm(Dk,'fro'); 
    Dk               = Dnew;
    counterr         = counterr + 1;
    
    % stopping criteria
    if mod(counterr, 10) == 0
        if rknorm > mu*sknorm
            delta = deltaInc*delta;
        else
            if sknorm > mu*rknorm
                delta = delta/deltaDecr;
            end
        end
    end
    if rknormR < epsPri && sknormR < epsDual
        stop = 1;
    end
    if counterr > maxIters
        stop = 1;
    end
    if Bnorm <1e-16
        stop = 1;
        Bnew = zeros(p,p);
        Dnew = zeros(p,p);
    end
end

%% Optimial value
DlastVec   = reshape(Dnew, [p^2,1]);
DlastVecU  = DlastVec(idxs);
MDlast     = U'*DlastVecU.*Sdiag;
optVal     = 0.5*norm( y - Vt'*MDlast )^2 + lambdaL*sum(abs(Dnew(:)));

%% Outputs
out         = struct;
out.optVal  = optVal;
out.count   = counterr;
out.delta   = delta;
out.Blast   = Bnew;
out.Dlast   = Dnew;
out.B       = Dnew;

end