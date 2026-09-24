import os
import numpy as np
import matplotlib.pyplot as plt


# The experiment results for each dataset
RESULTS_DIRS = {
    'MovieLens-20M': './results_movielens/observed_only_results_20260921_160522/npy',
    'Last.fm':       './results_lastfm/observed_only_results_20260921_212817/npy',
    'Amazon':        './results_amazon/observed_only_results_20260921_212827/npy',
}

# Toggle the two structured baselines independently.
PLOT_STRUCTURED_BASELINES = {
    'CoFineUCB': True,
    'H2N_Bandit': True,
}

# Structured baselines are stored separately from the matched flat-backbone
# and TDGE runs. Set a dataset path to None until its result arrays are ready.
STRUCTURED_RESULTS_DIRS = {
    'CoFineUCB': {
    'MovieLens-20M': './results_movielens/observed_only_results_20260921_160522/npy',
    'Last.fm':       './results_lastfm/observed_only_results_20260921_212817/npy',
    'Amazon':        './results_amazon/observed_only_results_20260921_212827/npy',
    },
    'H2N_Bandit': {
    'MovieLens-20M': './results_movielens/observed_only_results_20260921_160522/npy',
    'Last.fm':       './results_lastfm/observed_only_results_20260921_212817/npy',
    'Amazon':        './results_amazon/observed_only_results_20260921_212827/npy',
    },
}

SAVE_DIR = './paper_figures'

# Plot the stored across-run standard deviations directly. The archived arrays
# therefore reproduce the uncertainty bands used in the paper figures.


# ============================================================================
# Visualization parameters
# ============================================================================
FONT_SCALE = 2.0   
SAVE_DPI   = 300   
SHARED_X_LABEL_SIZE = 28
SHARED_Y_LABEL_SIZE = 24  # Twice the default shared-label size of 12 pt.

plt.rcParams.update({
    'xtick.labelsize':  10 * FONT_SCALE,
    'ytick.labelsize':  10 * FONT_SCALE,
    'axes.labelsize':   12 * FONT_SCALE,
    'axes.titlesize':   14 * FONT_SCALE,
    'legend.fontsize':  11 * FONT_SCALE,
    'figure.titlesize': 16 * FONT_SCALE,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'serif',
})


ALGORITHMS = [
    # ---- Linear family ----
    {'key': 'LinUCB',                'label': 'LinUCB',
     'color': '#F5A66B', 'linestyle': '--', 'family': 'linear'},
    {'key': 'LowD_HieRec_LinearUCB', 'label': 'TDGE-LinUCB',
     'color': '#D94E1F', 'linestyle': '-',  'family': 'linear'},

    # ---- Structured baselines ----
    {'key': 'CoFineUCB',             'label': 'CoFineUCB',
     'color': '#8E5AA9', 'linestyle': ':',  'family': 'structured'},
    {'key': 'H2N_Bandit',            'label': r'H$_2$N-Bandit',
     'color': '#555555', 'linestyle': '-.', 'family': 'structured'},

    # ---- NeuralUCB family ----
    {'key': 'NeuralUCB',             'label': 'NeuralUCB',
     'color': '#6EB8E0', 'linestyle': '--', 'family': 'neuralucb'},
    {'key': 'LowD_HieRec_NeuralUCB', 'label': 'TDGE-NeuralUCB',
     'color': '#1A5C99', 'linestyle': '-',  'family': 'neuralucb'},

    # ---- NeuralTS family ----
    {'key': 'NeuralTS',              'label': 'NeuralTS',
     'color': '#86D47A', 'linestyle': '--', 'family': 'neuralts'},
    {'key': 'LowD_HieRec_NeuralTS',  'label': 'TDGE-NeuralTS',
     'color': '#2E8B2E', 'linestyle': '-',  'family': 'neuralts'},
]


LEGEND_ORDER = [
    # Matplotlib fills multi-row legends column by column. This order produces
    # one visual row for each backbone family when ncol=3.
    'LinUCB', 'NeuralUCB', 'NeuralTS',
    'CoFineUCB', r'H$_2$N-Bandit', 'TDGE-NeuralTS',
    'TDGE-LinUCB', 'TDGE-NeuralUCB',
]

SIMPLE_LEGEND_ORDER = [
    'LinUCB', 'NeuralUCB', 'NeuralTS',
    'TDGE-LinUCB', 'TDGE-NeuralUCB', 'TDGE-NeuralTS',
]

FIGURES = [
    {
        'save_name':    'fig_online_training_regret',
        'y_label':      'Cumulative Regret',
        'x_label':      'Rounds',
        'mean_pattern': 'mean_regret_{alg}.npy',
        'std_pattern':  'std_regret_{alg}.npy',
        'style':        'training',
    },
    {
        'save_name':    'fig_cold_start_cumulative_regret',
        'y_label':      'Cumulative Regret',
        'x_label':      'Interaction Steps (per User)',
        'mean_pattern': 'mean_cs_cum_regret_{alg}.npy',
        'std_pattern':  'std_cs_cum_regret_{alg}.npy',
        'style':        'coldstart',
    },
    {
        'save_name':    'fig_cold_start_average_regret',
        'y_label':      'Average Regret',
        'x_label':      'Interaction Steps (per User)',
        'mean_pattern': 'mean_cs_avg_regret_{alg}.npy',
        'std_pattern':  'std_cs_avg_regret_{alg}.npy',
        'style':        'coldstart',
    },
    {
        'save_name':    'fig_cold_start_average_reward',
        'y_label':      'Average Reward',
        'x_label':      'Interaction Steps (per User)',
        'mean_pattern': 'mean_cs_reward_{alg}.npy',
        'std_pattern':  'std_cs_reward_{alg}.npy',
        'style':        'coldstart',
    },
]

def load_metric(results_dir, alg_key, fig_cfg):
    mean_path = os.path.join(results_dir, fig_cfg['mean_pattern'].format(alg=alg_key))
    std_path  = os.path.join(results_dir, fig_cfg['std_pattern'].format(alg=alg_key))
    if not os.path.exists(mean_path):
        return None, None
    mean = np.load(mean_path)
    std  = np.load(std_path) if os.path.exists(std_path) else None
    return mean, std


def plot_panel(ax, results_dir, dataset_name, fig_cfg, is_first_col):
    is_cs = (fig_cfg['style'] == 'coldstart')
    last_valid_len = None

    for cfg in ALGORITHMS:
        metric_dir = results_dir
        if cfg['family'] == 'structured':
            if not PLOT_STRUCTURED_BASELINES.get(cfg['key'], False):
                continue
            metric_dir = STRUCTURED_RESULTS_DIRS.get(cfg['key'], {}).get(dataset_name)
            if metric_dir is None:
                print(f"No results directory configured for [{dataset_name}] "
                      f"{cfg['label']} — skip")
                continue

        mean, std = load_metric(metric_dir, cfg['key'], fig_cfg)
        if mean is None:
            print(f"Can not find [{dataset_name}] {fig_cfg['mean_pattern'].format(alg=cfg['key'])} — skip")
            continue
        last_valid_len = len(mean)

        # Downsample long online trajectories for compact vector figures while
        # retaining the first and final observations. Cold-start trajectories
        # contain only ten points and are plotted in full.
        if not is_cs and len(mean) > 1000:
            plot_idx = np.unique(np.linspace(0, len(mean) - 1, 1000, dtype=int))
        else:
            plot_idx = np.arange(len(mean))
        x = plot_idx + 1
        mean_plot = mean[plot_idx]
        plot_kwargs = dict(
            label=cfg['label'],
            color=cfg['color'],
            linestyle=cfg['linestyle'],
            linewidth=2.5,
            zorder=3,
        )
  
        if is_cs:
            plot_kwargs.update(marker='o', markersize=6, markeredgewidth=0)

        ax.plot(x, mean_plot, **plot_kwargs)

        if std is not None:
            std_plot = std[plot_idx]
            ax.fill_between(x, mean_plot - std_plot, mean_plot + std_plot,
                            color=cfg['color'], alpha=0.15,
                            linewidth=0, zorder=2)

    ax.set_title(dataset_name, fontweight='bold', pad=10)

    if is_cs:
        if last_valid_len is not None:
            ax.set_xticks(np.arange(1, last_valid_len + 1))
   

    ax.grid(True, linestyle=':', alpha=0.5, zorder=1)


def build_shared_legend(fig, ax_for_handles):
    handles_by_label = {h.get_label(): h for h in ax_for_handles.get_lines()}
    ordered_handles, ordered_labels = [], []
    has_structured = any(
        lbl in handles_by_label for lbl in ('CoFineUCB', r'H$_2$N-Bandit')
    )
    legend_order = LEGEND_ORDER if has_structured else SIMPLE_LEGEND_ORDER
    for lbl in legend_order:
        if lbl in handles_by_label:
            ordered_handles.append(handles_by_label[lbl])
            ordered_labels.append(lbl)

    fig.legend(ordered_handles, ordered_labels,
               loc='lower center',
               ncol=3 if len(ordered_labels) > 6 else len(ordered_labels),
               bbox_to_anchor=(0.5, 0.04),
               frameon=False,
               handlelength=2.2,
               columnspacing=1.8)


def make_figure(fig_cfg, results_dirs, save_dir):
    n = len(results_dirs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (dname, rdir) in enumerate(results_dirs.items()):
        plot_panel(axes[i], rdir, dname, fig_cfg, is_first_col=(i == 0))

    build_shared_legend(fig, axes[0])
    fig.supxlabel(fig_cfg['x_label'], y=0.35,
                  fontsize=SHARED_X_LABEL_SIZE)
    fig.supylabel(fig_cfg['y_label'], x=0.04, y=0.65,
                  fontsize=SHARED_Y_LABEL_SIZE)

    plt.tight_layout(rect=(0.035, 0.34, 1, 1))
    plt.subplots_adjust(wspace=0.22)

    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{fig_cfg['save_name']}.pdf")
    png_path = os.path.join(save_dir, f"{fig_cfg['save_name']}.png")
    plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.savefig(png_path, dpi=SAVE_DPI, bbox_inches='tight')
    print(f"✅ saved: {pdf_path}")
    print(f"✅ saved: {png_path}")
    plt.close(fig)


def main():
    for fig_cfg in FIGURES:
        print(f"\n--- Plotting {fig_cfg['save_name']} ({fig_cfg['y_label']}) ---")
        make_figure(fig_cfg, RESULTS_DIRS, SAVE_DIR)


if __name__ == '__main__':
    main()
