# VHEM_H3M

_Reasonably fast_, _mostly correct_, implementation of the VHEM-H3M (Variational Hierarchical EM - Hidden Markov Mixture Model) [1] for clustering of HMMs (Hidden Markov Models).

This is mostly a translation of the equations to Julia code, with the use of log-values to improve numerical stability. It supports HMMs with GMM (Gaussian Mixture Model) emissions. The number of states, and of components per state, can be different between the HMMs. HMMs are specified using the [HMMBase](https://github.com/maxmouchet/HMMBase.jl) package.

This works reasonably well but there is room for improvement:
- Initialization of the reduced H3M (something like K-means++ ?)
- More in-place operations to reduce allocations
- Proper separation of the E and M steps
- Structure for summary statistics
- Documentation, examples, tests ...
- Profiling, optimization, cleanup ...

[1] Coviello, E., Chan, A. B., & Lanckriet, G. R. (2014). Clustering hidden Markov models with variational HEM. _The Journal of Machine Learning Research_, 15(1), 697-747. http://jmlr.org/papers/volume15/coviello14a/coviello14a.pdf

## Usage

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> registry add https://github.com/maxmouchet/JuliaRegistry.git
pkg> add VHEM_H3M
```

```julia
using Distributions, HMMBase, VHEM_H3M

# Simulate a dataset of HMMs
randgmm(K) = MixtureModel([Normal(rand(Normal(0, 10)), 1) for _ in 1:K])
randhmm(K) = HMM(randtransmat(K), [randgmm(rand(1:5)) for _ in 1:K])
base_models = [randhmm(rand(1:10)) for _ in 1:100]

# Initialization
base = H3M(base_models)
redu = H3M(rand(base_models, 2))

# Clustering
τ = 10   # Length of the virtual sequences
N = 1000 # Number of virtual samples

hist, z, reducedp = cluster(base, redu, τ, N)

# hist: EM history (convergence, logtots)
# z: optimal assignments (base x reduced)
# reducedp: final reduced models

# To get an hard clustering (of the base models with respect to the reduced models),
# we can take the argmax of z:
labels = [x.I[2] for x in argmax(z, dims = 2)]
```

## Development

```
└── src
    ├── api.jl       # High-level interface (`cluster`)
    ├── em.jl        # Variational E and M steps
    ├── h3m.jl       # H3M type
    ├── lse.jl       # Streaming log-sum-exp implementation
    ├── mc.jl        # Monte-Carlo expectations (not used)
    ├── va.jl        # Variational expectations
    └── VHEM_H3M.jl  # Module definition
```
