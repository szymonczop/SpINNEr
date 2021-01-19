% Proximity operator for function G, where
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%                  G(C) := lambda_N*||C||_*                            %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ||C||_* - nuclear norm of C
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%         Author:    Damian Brzyski
%         Date:      April 24, 2018
%-------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
function Cnew = ProxG(D, W2, delta, lambda_N)

Ddelta    = D + W2/delta;
Ddelta    = (Ddelta + Ddelta')/2;
[U, S, V] = eig(Ddelta);  % Ddelta = U*S*V';
diagS     = diag(S);
Stsh      = diag(  sign(diagS).*max(abs(diagS) - lambda_N/delta, 0)  ); % soft tresholoding of singular values
Cnew      = U*Stsh*V';

end
