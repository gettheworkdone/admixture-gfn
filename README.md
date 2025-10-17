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
  Wishart-based likelihood components from AdmixtureBayes and now contains
  fully documented helper routines, bootstrap degree-of-freedom estimation,
  and verbose logging to surface every step of the reward calculation.【F:src/gfnx/reward/admixture.py†L1-L418】
* **Policies** – instead of hand-crafted proposal kernels, policies are neural
  networks (or simpler baselines) trained with a trajectory-balance objective
  that jointly fits forward/backward policies and the log-partition constant of
  the target reward.【F:baselines/dummy_admixture.py†L62-L214】

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

1. Clone the repository and create the conda environment defined in
   `envs/admixture-gfn.yaml`:

   ```bash
   conda env create -f envs/admixture-gfn.yaml
   conda activate admixture-gfn
   ```

   The specification targets CPU-only execution (`jax[cpu]`) so it works on
   machines without a GPU. Feel free to swap the JAX wheel in the YAML file for
   a CUDA/ROCm build if acceleration is available.

2. Add the local sources to `PYTHONPATH` (or install `gfnx` as a package) so the
   baseline scripts can import the environment modules:

   ```bash
   export PYTHONPATH=$PWD/src
   ```

3. (Optional) Install developer tooling such as `pytest` for the unit tests.

## Quick start

The `baselines/dummy_admixture.py` script demonstrates end-to-end usage. It
initialises the GFlowNet environment, instantiates the reward module (printing
out the resolved Arctic SNP dataset and bootstrap statistics), and trains a
small MLP policy with the trajectory-balance loss. Training alternates between
configurable blocks of optimisation steps and validation rollouts while writing
rich console diagnostics and TensorBoard scalars. Launch it with:

```bash
python baselines/dummy_admixture.py
```

Hydra reads configuration from `baselines/configs/dummy_admixture.yaml`, so you
can override arguments via the command line—for example:

* `training.train_steps_per_phase=10` to control how many gradient steps happen
  between validation passes.
* `validation.num_envs=16` to change how many graphs are sampled during each
  evaluation block.
* `policy.hidden_size=256 policy.depth=3` to widen/deepen the MLP without
  touching the source code.

The training loop emits detailed per-step diagnostics via `jax.debug.print`,
while the Python logger surfaces environment and optimiser metadata before
training begins.【F:baselines/dummy_admixture.py†L124-L214】 TensorBoard traces
are written under `tensorboard/` in the Hydra run directory; point TensorBoard
at that folder to inspect loss/reward curves:

```bash
tensorboard --logdir tensorboard
```

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
