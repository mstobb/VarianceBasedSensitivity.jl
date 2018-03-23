# VarianceBasedSensitivity

[![Build Status](https://travis-ci.org/mstobb/VarianceBasedSensitivity.jl.svg?branch=master)](https://travis-ci.org/mstobb/VarianceBasedSensitivity.jl)

[![Coverage Status](https://coveralls.io/repos/mstobb/VarianceBasedSensitivity.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mstobb/VarianceBasedSensitivity.jl?branch=master)

[![codecov.io](http://codecov.io/github/mstobb/VarianceBasedSensitivity.jl/coverage.svg?branch=master)](http://codecov.io/github/mstobb/VarianceBasedSensitivity.jl?branch=master)


This is a set of functions to generate and compute variance based sensitivity
metrics (usually called Sobol Indices) in Julia.  The sampling method is used
with a user supplied function.  For large or complex functions, an emulator is
typically used, but this is not implemented here (i.e. construct your own
emulator).

To install, use the command:
```julia
Pkg.clone("git:github.com/mstobb/VarianceBasedSensitivity.jl.git")
```
which will clone the repository and install.

Typical usage might look the following:
```julia
using VarianceBasedSensitivity, Distributions

# User supplied function
f(x) = [x[1]+x[2]+x[3]; x[1]*x[2]*x[3]]

# Set the distribution of the parameters
sob = SobolDists(3,[Uniform(0.0,10.0),Uniform(-2.0,2.0),Uniform(10.0,20.0)]) 

# Obtain 100 raw sobol samples (does not evaluate function)
sobsamp = makeSobolSamples(sob,100);

# Evaluate function at specified sample points
sobevals = sobolSampler(f,sobsamp)

# Compute just the first and 2nd order sensitivities
sobEst = computeSensSij(sobevals)

# Compute the above estimates and the uncertainty in them using Bootstrap
sobEstU = sensSijSTD(sobevals)
```

