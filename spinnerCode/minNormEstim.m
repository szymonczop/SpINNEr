function out = minNormEstim(y, SVDAx)

%% Objects
p0   = size(SVDAx.U,1);
p    = (1+sqrt(1+8*p0))/2;

%% SVD outputs 
U      = SVDAx.U;
Sdiag  = SVDAx.Sdiag;
Vt     = SVDAx.Vt;
idxs   = SVDAx.idxs;

%% Estimate
Vty            = Vt * y./Sdiag;
Bvec           = U*Vty;
Bestim         = zeros(p^2, 1);
Bestim(idxs,:) = Bvec; 
Bestim         = reshape(Bestim, [p, p]); % p-by-p matrix
Bestim         = Bestim + Bestim';
out            = struct;
out.B          = Bestim;

end
