# TDGE reproducibility package

This directory contains the minimal code and final artifacts needed to
reproduce the recommendation experiments and figures reported in
**Human-inspired, Task-Dimension-Guided Exploration for Efficient Learning in
High Dimensions**. The original development directory is not required.

## Contents

```text
.
├── preprocess_{movielens,lastfm,amazon}.py   # dataset preprocessing
├── train_{movielens,lastfm,amazon}.py        # training and reported ablations
├── alg.py, neuralucb.py, env.py              # contextual-bandit components
├── alg_runner.py, ablation_no_fd.py          # TDGE runner and NoFD ablation
├── cofine_ucb.py, h2n_bandit.py              # structured baselines
├── item_cluster_bandit.py                    # item-clustering ablation
├── tdge_multilevel.py                        # hierarchy-depth ablation
├── compute_final_metrics_cross_runs.py       # main metrics and Welch tests
├── compute_final_metrics_clustering.py       # clustering-ablation metrics
├── plot_final.py                             # paper curves
├── results_*/.../npy/                        # final aggregate/run arrays
├── paper_figures/                            # final tables and figures
├── *_metric_config.json                      # result-directory mappings
└── scripts/reproduce_reported_results.sh     # metrics and plots entry point
```

Only result arrays used by the reported metrics and plots are retained.
Per-user diagnostic traces, checkpoints, exploratory analyses, parameter
sweeps, and development utilities are omitted.

## Environment

The experiments used Linux, Python 3.10, and the versions in
`requirements.txt`. A CUDA-capable GPU is recommended for neural methods.

```bash
conda create -n tdge python=3.10 -y
conda activate tdge
pip install -r requirements.txt
```

The preprocessing scripts download
`sentence-transformers/all-mpnet-base-v2` on first use.

## Reproduce the reported metrics and plots

The final aggregate results are included. From this directory, run:

```bash
bash scripts/reproduce_reported_results.sh
```

This command recomputes the paper metrics, two-sided Welch tests, clustering
ablation summary, and all regret/reward curves. Outputs are written to
`paper_figures/`.

The paper and archived result files report population standard deviations
(`ddof=0`) across the five runs. The metric scripts preserve those displayed
values and convert them to sample standard deviations internally when
performing Welch tests. The plotting script uses the archived standard
deviations directly, reproducing the paper's uncertainty bands.

## Datasets

Raw datasets are not redistributed. Download and extract them into the
following layout:

```text
datasets/
├── ml-20m/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── genome-scores.csv
│   └── genome-tags.csv
├── lastfm/
│   ├── artists.dat
│   ├── user_artists.dat
│   ├── tags.dat
│   └── user_taggedartists.dat
└── Amazon_2023/
    ├── Beauty_and_Personal_Care.jsonl.gz
    └── meta_Beauty_and_Personal_Care.jsonl.gz
```

Sources:

- MovieLens-20M: <https://files.grouplens.org/datasets/movielens/ml-20m.zip>
- Last.fm HetRec 2011: <https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip>
- Amazon Reviews 2023: <https://amazon-reviews-2023.github.io/>

## Preprocessing

```bash
python preprocess_movielens.py \
  --clustering_method hierarchical \
  --data_dir datasets/ml-20m_data

python preprocess_lastfm.py \
  --clustering_method hierarchical \
  --data_dir datasets/lastfm_data

python preprocess_amazon.py \
  --clustering_method hierarchical \
  --data_dir datasets/amazon_data
```

The same scripts support `kmeans`, `spectral`, and `gmm` for the clustering
ablation. Use a separate `--data_dir` for each method.

## Training

The following commands reproduce the five-run experiments with seeds
2026--2030, 10,000 online rounds, a 90/10 train/cold-start split, 100 held-out
users, and ten cold-start interactions per user:

```bash
TDGE_NUM_RUNS=5 TDGE_NUM_STEPS=10000 \
TDGE_COLD_START_USERS=100 TDGE_COLD_START_STEPS=10 \
python train_movielens.py

TDGE_NUM_RUNS=5 TDGE_NUM_STEPS=10000 \
TDGE_COLD_START_USERS=100 TDGE_COLD_START_STEPS=10 \
python train_lastfm.py

TDGE_NUM_RUNS=5 TDGE_NUM_STEPS=10000 \
TDGE_COLD_START_USERS=100 TDGE_COLD_START_STEPS=10 \
python train_amazon.py
```

To run selected methods only, provide comma-separated internal method names:

```bash
TDGE_ALGORITHMS=NeuralUCB,LowD_HieRec_NeuralUCB \
TDGE_NUM_RUNS=5 python train_movielens.py
```

The training scripts include the flat backbones, TDGE variants, CoFineUCB,
H2N-Bandit, NoFD, item-clustering, and hierarchy-depth ablations. Each run
creates a timestamped directory under `results_<dataset>/`. Point
`cross_run_metric_config.json` and `plot_final.py` to a new result directory
before recomputing metrics and figures.

## Metric and plotting commands

```bash
python compute_final_metrics_cross_runs.py \
  --config cross_run_metric_config.json \
  --save-dir paper_figures \
  --num-runs 5 \
  --saved-std-ddof 0

python compute_final_metrics_clustering.py \
  --config clustering_metric_config.json \
  --save-dir paper_figures \
  --num-runs 5 \
  --saved-std-ddof 0

python plot_final.py
```

