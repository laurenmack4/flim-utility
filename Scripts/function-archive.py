import numpy as np
import matplotlib.pyplot as plt

# functions that i made and then did not need in the main script
# but am keeping just in case

def raincloud_plot(lifetime_data):
    """
    function to make and display a raincloud plot of fitted lifetime values
    """
    # selection of nonzero lifetimes (those =0 are below threshold)
    bright_tau1 = lifetime_data.t0_image[lifetime_data.tcspc_image.intensity_image > 1 * np.median(lifetime_data.tcspc_image.intensity_image)]
    tau1 = np.asarray(bright_tau1).ravel() * 1e9
    tau1 = tau1[tau1 != 0]

    fig, ax = plt.subplots(figsize=(6, 5))

    # left: violin half plot
    vp = ax.violinplot([tau1], positions=[1], widths=1, showextrema=False)
    vertices = vp["bodies"][0].get_paths()[0].vertices
    vertices[:, 0] = np.minimum(vertices[:, 0], 1)  # keep the left half
    vp["bodies"][0].set_facecolor("thistle")
    vp["bodies"][0].set_alpha(0.7)

    # centre: box plot
    bp = ax.boxplot([tau1], positions=[1], widths=0.12, patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("yellowgreen")
    bp["boxes"][0].set_alpha(0.6)

    # right: values, jittered
    x = 1.15 + np.random.uniform(-0.05, 0.05, size=len(tau1))
    ax.scatter(x, tau1, s=8, c="tomato", alpha=0.6, marker='.', label=f'mean={np.mean(tau1):.3g} ns\n SD={np.std(tau1):.3g} ns')

    ax.set_xlim(0.45, 1.65)
    ax.set_xticks([1], ["Tau 1"])
    ax.set_ylabel("Lifetime (ns)")
    ax.set_title("Non-zero lifetimes raincloud plot")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()