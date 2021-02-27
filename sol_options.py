

class Options:
    deltaInitial1   =  100   # the initial "step length" for the update with nuclear norm (i.e. delta1)
    deltaInitial2 = 100 # the initial "step length" for the update with LASSO norm (i.e.delta2)
    scaleStep = 1#  the initial scale for updated deltas; the scale is changed in repetitions based on the convergence rates
    ratioStep = 1 # the initial ratio between updated deltas; the ratio is changed in repetitions based on the convergence rates
    mu = 10 # the maximal acceptable ratio between convergence rates to keep deltas without changes in next iteration
    deltaInc = 2 # delta is multiplied by this parameter when the algorithm decides that it should be increased
    deltaDecr = 2 # delta is divided by this parameter when the algorithm decides that it should be decreased
    ratioInc = 2 # ratio is multiplied by this parameter when the algorithm decides that it should be increased
    ratioDecr = 2 # ratio is divided by this parameter when the algorithm decides that it should be decreased
    maxIters = 50000 # the maximal number of iterations; this is a stopping criterion if the algorithm does not converge
    epsPri = 1e-6 # convergence tolerance, primar residual
    epsDual = 1e-6 # convergence tolerance, dual residual
