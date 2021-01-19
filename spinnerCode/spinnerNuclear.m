function out = spinnerNuclear(y, SVDAx, lambda_N, solOptions)

% this function solves the problem 
%
% argmin_B {  0.5*sum_i (y_i - <A_i, B>)^2 + lambda_N || B ||_*  }
% 
% y and AA are after regressing X out

%% Objects
p0           = size(SVDAx.U,1);
p            = (1+sqrt(1+8*p0))/2;

%% Solver options
deltaInitial  = solOptions.deltaInitial1;
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
Ck = zeros(p,p);
Wk = zeros(p,p);

%% ADMM loop
delta    = deltaInitial;
counterr = 0;
stop     = 0;
while stop == 0
    Bnew = ProxFsvd(y, SVDAx, Ck, Wk, delta);
    Cnew = ProxG(Bnew, -Wk, delta, lambda_N);
    Wk   = Wk + delta*(Cnew - Bnew);  
    rk               = Cnew - Bnew;
    sk               = Cnew - Ck;
    rknorm           = norm(rk,'fro');
    Bnorm            = norm(Bnew,'fro');
    rknormR          = rknorm/Bnorm;
    sknorm           = norm(sk,'fro');
    sknormR          = sknorm/norm(Ck,'fro'); 
    Ck               = Cnew;
    counterr         = counterr + 1;
    
    % stopping criteria
    if counterr > 10
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
        Cnew = zeros(p,p);
    end
end

%% Optimial value
ClastVec   = reshape(Cnew, [p^2,1]);
ClastVecU  = ClastVec(idxs);
MClast     = U'*ClastVecU.*Sdiag;
optVal     = 0.5*norm( y - Vt'*MClast )^2 + lambda_N*sum(svd(Cnew));

%% Outputs
out         = struct;
out.optVal  = optVal;
out.count   = counterr;
out.delta   = delta;
out.Blast   = Bnew;
out.Clast   = Cnew;
out.B       = Cnew;

end