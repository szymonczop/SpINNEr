% Proximity operator for function H, where
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%                  H(D) := lambda2*|| vec(D) ||_1                     %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      April 24, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 
function Dnew = ProxH(B, C, delta1, delta2, W1, W2, lambda_L, WGTs)
    deltas     = delta1 + delta2;
    Bdelta1    = B - W1/delta1;
    Bdelta2    = C - W2/delta2;
    Bdelta     = (delta1*Bdelta1 + delta2*Bdelta2)/deltas;
    Dnew       = sign(Bdelta).*max(abs(Bdelta) - WGTs*lambda_L/deltas, 0);
end