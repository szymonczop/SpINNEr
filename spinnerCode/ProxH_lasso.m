% Proximity operator for function H, where
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%                H(D) := lambdaL*|| vec(W o D) ||_1                   %%%
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
function Dnew = ProxH_lasso(B, delta, W, lambdaL, WGTs)
    Bdelta     = B - W/delta;
    Dnew       = sign(Bdelta).*max(abs(Bdelta) - WGTs*lambdaL/delta, 0);
end