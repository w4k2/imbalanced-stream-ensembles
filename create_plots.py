import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from medpy.filter.smoothing import anisotropic_diffusion

fnames = [
    "sudden_drift_5",
    "sudden_drift_10",
    "sudden_drift_20",
    "sudden_drift_30",
    "gradual_drift_5",
    "gradual_drift_10",
    "gradual_drift_20",
    "gradual_drift_30",
]

ftitles = {
    "sudden_drift_5": "Sudden drift with 5% of minority class",
    "sudden_drift_10": "Sudden drift with 10% of minority class",
    "sudden_drift_20": "Sudden drift with 20% of minority class",
    "sudden_drift_30": "Sudden drift with 30% of minority class",
    "gradual_drift_5": "Gradual drift with 5% of minority class",
    "gradual_drift_10": "Gradual drift with 10% of minority class",
    "gradual_drift_20": "Gradual drift with 20% of minority class",
    "gradual_drift_30": "Gradual drift with 30% of minority class",
}

for ind, fname in enumerate(fnames):
    results = np.load("scores_refit/Stream_%s_imbalance.npy" % fname)
    results_mse = np.load("scores_mse_2/Stream_%s_imbalance.npy" % fname)
    results_wo_samp = np.load("scores_weights/Stream_%s_imbalance.npy" % fname)
    new_results = np.concatenate((results, results_mse, results_wo_samp), axis=1)

    new_results = np.mean(new_results, axis=0)
    kernel = 1.5

    labels = [
        "u-AWE-G",
        "u-AWE-B",
        "u-AWE-F",
        "o-AWE-G",
        "o-AWE-B",
        "o-AWE-F",
        "u-AUE-G",
        "u-AUE-B",
        "u-AUE-F",
        "o-AUE-G",
        "o-AUE-B",
        "o-AUE-F",
        "WAE",
        "OOB",
        "UOB",
        "AWE",
        "u-AWE",
        "o-AWE",
        "AUE",
        "u-AUE",
        "o-AUE",
        "AWE-G",
        "AWE-B",
        "AWE-F",
        "AUE-G",
        "AUE-B",
        "AUE-F",
    ]

    # Comparision of proposed models, one metric per figure

    metrics = ["f-score", "gmean", "bac"]
    for m_i, m in enumerate(metrics):
        titles = [
            "models with modified weights",
            "models with oversampling",
            "models with undersampling",
        ]
        colors = [
            [
                (1, 0.8, 0.8),
                (0.8, 0.8, 1),
                "red",
                "blue",
                "red",
                "blue",
                "red",
                "blue",
            ],
            [
                (1, 0.7, 0.7),
                (0.7, 0.7, 1),
                (1, 0.7, 0.7),
                (0.7, 0.7, 1),
                "red",
                "blue",
                "red",
                "blue",
                "red",
                "blue",
            ],
            [
                (1, 0.7, 0.7),
                (0.7, 0.7, 1),
                (1, 0.7, 0.7),
                (0.7, 0.7, 1),
                "red",
                "blue",
                "red",
                "blue",
                "red",
                "blue",
            ],
        ]
        ls = [
            ["-", "-", ":", ":", "--", "--", "-.", "-."],
            ["-", "-", ":", ":", ":", ":", "--", "--", "-.", "-."],
            ["-", "-", ":", ":", ":", ":", "--", "--", "-.", "-."],
        ]

        usages = [
            [15, 18, 21, 24, 22, 25, 23, 26],
            [15, 18, 17, 20, 3, 9, 4, 10, 5, 11],
            [15, 18, 16, 19, 0, 6, 1, 7, 2, 8],
        ]
        lw = [
            [1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
            [1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
            [1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
        ]

        locs = [1, 3, 3]

        print(new_results.shape)

        fig, ax = plt.subplots(3, 1, figsize=(8, 11))
        for i in range(3):
            for j, row in enumerate(new_results[usages[i], :, m_i]):
                # res = anisotropic_diffusion(row, gamma=0.05, kappa=300, niter=10)
                res = gaussian_filter(row, kernel)
                ax[i].plot(
                    res, c="white", ls="-", lw=lw[i][j] + 2,
                )
                ax[i].plot(
                    res,
                    c=colors[i][j],
                    ls=ls[i][j],
                    label=labels[usages[i][j]],
                    lw=lw[i][j],
                )
            ax[i].set_title(titles[i])
            ax[i].set_ylim(0, 1)
            ax[i].set_xlim(0, 100 - 1)
            ax[i].legend(ncol=len(usages[i]) // 2, frameon=False)
            ax[i].spines["right"].set_color("none")
            ax[i].spines["top"].set_color("none")
            ax[i].grid(ls=":")

        fig.suptitle(ftitles[fname] + " for " + m + " metric.")
        fig.subplots_adjust(top=0.93, bottom=0.03, left=0.06, right=0.97)

        # plt.tight_layout()
        plt.savefig("foo.png")
        plt.savefig("figures/%s.png" % (fname + " " + m))

        # exit()

    # Comparision of selected (the best) proposed models with WAE, OOB and UOB
    colors = [
        ["dodgerblue", "cyan", "darkviolet", "mediumslateblue", "green", "blue", "red"],
        ["dodgerblue", "cyan", "darkviolet", "mediumslateblue", "green", "blue", "red"],
        ["dodgerblue", "cyan", "darkviolet", "mediumslateblue", "green", "blue", "red"],
    ]

    usages = [
        [9, 10, 11, 26, 12, 13, 14],
        [9, 10, 11, 26, 12, 13, 14],
        [9, 10, 11, 26, 12, 13, 14],
    ]

    lw = [
        [2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 1, 1, 1],
    ]
    locs = [1, 3, 3]

    print(new_results.shape)

    fig, ax = plt.subplots(3, 1, figsize=(8, 11))
    for i in range(3):
        for j, row in enumerate(new_results[usages[i], :, i]):
            res = gaussian_filter(row, kernel)
            ax[i].plot(
                res, c=colors[i][j], ls="-", label=labels[usages[i][j]], lw=lw[i][j]
            )
        ax[i].set_title(metrics[i])
        ax[i].set_ylim(0, 1)
        ax[i].legend(ncol=3, frameon=False)

    fig.suptitle(fname)
    # plt.tight_layout()
    plt.savefig("bar.png")
    plt.savefig("figures/%s.png" % fname)
    # exit()
