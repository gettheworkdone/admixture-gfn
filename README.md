# Admixture-GFN

Admixture-GFN experiments with Generative Flow Networks (GFlowNets) to sample
phylogenetic admixture graphs. The project draws on the inference pipeline from
[AdmixtureBayes](https://github.com/avaughn271/AdmixtureBayes)—which relies on a
Markov chain Monte Carlo (MCMC) sampler—and replaces the sampling procedure with
GFlowNet-based rollouts and policies. The goal is to maintain compatibility with
the allele count inputs and posterior analyses familiar to AdmixtureBayes users
while exploring a learning-based approach to exploring the graph space.

## How this differs from AdmixtureBayes

AdmixtureBayes follows a three-step workflow (`runMCMC`, `analyzeSamples`, and
`makePlots`) to build posterior samples of admixture graphs via tempered MCMC.
This repository keeps the same high-level structure but targets a different
sampling core:

* **Trajectory generation** – `gfnx.environment.admixture.AdmixtureGraphEnvironment`
  defines a vectorized JAX environment that constructs admixture graphs by
  applying forward and backward actions over adjacency matrices instead of the
  explicit MCMC proposals used in AdmixtureBayes.【F:src/gfnx/environment/admixture.py†L16-L120】
* **Rewards** – `gfnx.reward.admixture.AdmixtureGraphRewardModule` ports the
  Wishart-based likelihood components from AdmixtureBayes so the learned
  policies optimise the same target distribution without running an MCMC
  chain.【F:src/gfnx/reward/admixture.py†L1-L140】
* **Policies** – instead of hand-crafted proposal kernels, policies are neural
  networks (or simpler baselines) trained to sample high-reward graphs via GFlowNet
  objectives.【F:baselines/dummy_admixture.py†L1-L109】

The accompanying AdmixtureBayes repository is the canonical reference for the
statistical model, input file format, and convergence diagnostics; this project
focuses on swapping the sampling machinery.

## Repository layout

```
baselines/        # Hydra-configured experiments and simple policy baselines
src/gfnx/         # Core GFlowNet environment, reward modules, networks, and utils
tests/            # Property and rollout tests for the environments
```

Key modules worth inspecting are `src/gfnx/base.py` for the abstract environment
API, `src/gfnx/utils/__init__.py` for rollout helpers, and `src/gfnx/networks/`
for policy architectures.

## Installation

1. Clone the repository and create a Python >=3.10 environment.
2. Install dependencies. A minimal setup looks like:

   ```bash
   pip install "jax[cpu]" chex jaxtyping equinox hydra-core jax-tqdm
   pip install networkx matplotlib numpy pandas scipy
   ```

   Adapt the JAX installation command to target CUDA/ROCm wheels if desired.

3. (Optional) Install developer tooling such as `pytest` for the unit tests.

## Quick start

The `baselines/dummy_admixture.py` script demonstrates end-to-end usage. It
initialises the GFlowNet environment and reward module, defines a simple uniform
policy, and runs a short rollout loop to sample trajectories. Launch it with:

```bash
python baselines/dummy_admixture.py
```

Hydra reads configuration from `baselines/configs/dummy_admixture.yaml`, so you
can override arguments via the command line (e.g. `python baselines/dummy_admixture.py environment.num_leaves=6`).

## Testing

Unit tests live under `tests/`. Run the entire suite with:

```bash
pytest
```

## Roadmap

The current focus is on validating the GFlowNet environment against known
AdmixtureBayes behaviour, scaling up policy models beyond the dummy baseline,
and implementing training objectives that mirror the target posterior. Future
work will include improved logging/visualisation to parallel the original
`analyzeSamples` and `makePlots` utilities.
