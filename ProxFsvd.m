% Proximity operator for function F, where
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%             F(B) := sum_i (y_i - <A_i, B>)^2                        %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A - assumed to be 3-way tensor
% y - vector of observations
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      April 24, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set C instead of D for pure nuclear norm

% it solves problem 


%  argmin_B {  sum_i (y_i - <A_i, B>)^2 + delta || B - Dk - Wk/delta ||_F^2  }

function Bnew = ProxFsvd(y, SVDAx, Dk, Wk, delta)

%% 
p = size(Wk,1);

%%
AU     = SVDAx.U;
ASdiag = SVDAx.Sdiag;
AVt    = SVDAx.Vt;
idxs   = SVDAx.idxs;

%%
Ddelta     = Dk + Wk/delta;
DdeltaVec  = reshape(Ddelta, [p^2,1]);
DdeltaVecU = DdeltaVec(idxs);
Mdelta     = AU'*DdeltaVecU.*ASdiag;
ySubs      = AVt'*Mdelta;
ydelta     = y - ySubs;

%% Ridge solution from SVD
AVty       = AVt * ydelta;
MAVty      = (AVty.*ASdiag)./(ASdiag.^2 + 2*delta); % element-wise
BridgeVec  = AU * MAVty;

%% Reconstruct the symmetric matrix
Bridge         = zeros(p^2, 1);
Bridge(idxs,:) = BridgeVec; 
Bridge         = reshape(Bridge, [p, p]); % p-by-p matrix
Bridge         = Bridge + Bridge';

%% B update
Bnew           = Bridge + Ddelta;

end
