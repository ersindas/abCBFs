"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Adaptive Backup Control Barrier Functions.

© 2026 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator


class PaperPlotter:
    def plotter(
        x,
        u_act,
        intervening,
        u_p,
        env,
        latex_plots=True,
        save_plots=False,
        show_plots=True,
        u_plots=False,
    ):
        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        # TODO CHANGE THESE LATER
        bCBF_color = "#6a4c93"
        robust_color = "#dc4bdc"

        if not env.safety_flag:
            mode = "noFilter"
            traj_color = "red"
        elif env.vanilla_flag:
            mode = "vanilla"
            traj_color = bCBF_color
        elif env.robust_flag:
            mode = "robust"
            traj_color = robust_color

        dpi = 200
        n_fig_size = 7
        n = n_fig_size

        # Define constants and extract values
        xaxis_sz, legend_sz, ticks_sz = 25, 23, 25
        x1 = x[0, :]
        x2 = x[1, :]
        N = len(x1)
        t_span = np.arange(N) * env.del_t
        t_span_u = np.arange(u_act.shape[1] - 1) * env.del_t
        lblsize = legend_sz * 1.35
        alpha_set = 0.4
        lwp = 2.5

        lwp_main = 4

        # DEFINE COLORS
        obstacle_color = "#666666"
        obstacle_c_fill = "#E6E6E6"
        color_goal = "#AD307D"

        plt.figure(figsize=(2 * n, 0.9 * n), dpi=dpi)

        ##################### 2D PLOT #####################
        ax1 = plt.gca()
        ax1.axis("equal")
        ax1.set_xlabel(r"$x \text{ (m)}$", fontsize=xaxis_sz)
        # if env.vanilla_flag:
        #     ax1.set_ylabel(r"$z \text{ (m)}$", fontsize=xaxis_sz)
        # else:
        #     ax1.set_yticks([])
        ax1.set_ylabel(r"$z \text{ (m)}$", fontsize=xaxis_sz)
        ax1.xaxis.set_tick_params(labelsize=ticks_sz)
        ax1.yaxis.set_tick_params(labelsize=ticks_sz)

        ax1.set_aspect("equal", adjustable="box")
        ax1.fill_between(
            [-50, 50],
            -10,
            20,
            # color=[217 / 255, 242 / 255, 208 / 255],
            color = "#eeeded",
            alpha=alpha_set,
        )

        plt.vlines(
            [-5, env.px_min], env.pz_min, 10, colors=obstacle_color, linewidth=2.2
        )
        plt.hlines(env.pz_min, env.px_min, 10, colors=obstacle_color, linewidth=2.2)

        plt.fill_between([-10, env.px_min], 0, 10, color=obstacle_c_fill) 
        plt.fill_between(
            [env.px_min, 10], env.pz_min, -10, color=obstacle_c_fill
        )
        ax1.set_ylim([0, 9])
        ax1.set_xlim([-1, 6])

        # Plot orientations of quadrotor
        L = 0.6
        # if env.vanilla_flag:
        o_step_freq = 7
        o_freq = env.backup_save_N * 14
        h = 0.15
        L_blade = 0.1
        tstampBufy = 0.99

        for i in range(0, N, o_freq):
            # Compute end points of the orientation line
            x_start = x[0, i] - L / 2 * np.cos(x[2, i])
            z_start = x[1, i] + L / 2 * np.sin(x[2, i])
            x_end = x[0, i] + L / 2 * np.cos(x[2, i])
            z_end = x[1, i] - L / 2 * np.sin(x[2, i])

            # Draw orientation line
            plt.plot([x_start, x_end], [z_start, z_end], "k-", linewidth=2)
            plt.scatter(x[0, i], x[1, i], color="black", zorder=5)

            # if i == 0:
            #     ax1.text(
            #         x[0, i] + 0.12,
            #         x[1, i] * tstampBufy,
            #         "t \!=\! 0",
            #         fontsize=lblsize / 1.5,
            #     )

            # Draw orientation of props
            x_prop_del = h * np.sin(x[2, i])
            z_prop_del = h * np.cos(x[2, i])
            plt.plot(
                [x_start, x_start + x_prop_del],
                [z_start, z_start + z_prop_del],
                "k-",
                linewidth=2,
                zorder=10,
            )
            plt.plot(
                [x_end, x_end + x_prop_del],
                [z_end, z_end + z_prop_del],
                "k-",
                linewidth=2,
                zorder=10,
            )

            # Draw prop blades
            x_blade_del = L_blade / 2 * np.cos(x[2, i])
            z_blade_del = L_blade / 2 * np.sin(x[2, i])

            x_bladeL = x_start + x_prop_del
            z_bladeL = z_start + z_prop_del
            x_bladeR = x_end + x_prop_del
            z_bladeR = z_end + z_prop_del
            plt.plot(
                [x_bladeL - x_blade_del, x_bladeL + x_blade_del],
                [z_bladeL + z_blade_del, z_bladeL - z_blade_del],
                "k-",
                linewidth=2,
                zorder=10,
            )
            plt.plot(
                [x_bladeR - x_blade_del, x_bladeR + x_blade_del],
                [z_bladeR + z_blade_del, z_bladeR - z_blade_del],
                "k-",
                linewidth=2,
                zorder=10,
            )

        plt.plot(
            x[0, :],
            x[1, :],
            color=traj_color,
            linewidth=lwp_main,
            linestyle="-",
        )

        lw_backup = 1
        estimate_b_color = "#a9a9a9"

        def plot_centered_rectangle(
            center_x,
            center_y,
            width,
            height,
            ax,
            color,
            fillcolor="None",
        ):
            bottom_left_x = center_x - width / 2
            bottom_left_y = center_y - height / 2
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor=fillcolor,
            )
            ax.add_patch(rect)

        # Plot backup trajectories
        if env.backupTrajs:
            max_x_phi = []
            min_x_phi = []
            max_numBackup = len(env.backupTrajs)
            for i, xy in enumerate(zip(env.backupTrajs[::o_step_freq])):
                if i == 0:
                    if env.vanilla_flag:
                        label = r"$\boldsymbol{\varphi}_{\rm b}(\tau,\boldsymbol{x})" + "$"
                    elif env.robust_flag:
                        # label = r"$\boldsymbol{\phi}_{\rm s}(\tau,\boldsymbol{x})" + "$"
                        label = (
                            r"$\skew{4}\widehat{\boldsymbol{\varphi}}_{\rm b} (\tau,{\boldsymbol{x}})"
                            + "$"
                        )
                else:
                    label = None
                    # label2 = None

                if i < max_numBackup:
                    plt.plot(
                        xy[0][:, 0],
                        xy[0][:, 1],
                        color=estimate_b_color,
                        linewidth=lw_backup,
                        linestyle="--",
                        label=label,
                        zorder=2,
                    )
                    max_x_phi.append(max(xy[0][:, 0]))
                    min_x_phi.append(min(xy[0][:, 0]))

        plt.scatter(
            env.primaryParams["xg"],
            env.primaryParams["zg"],
            50,
            color=color_goal,
            label=r"$(x_{\rm g}, z_{\rm g})$",
        )
        plot_centered_rectangle(
            env.primaryParams["xg"],
            env.primaryParams["zg"],
            0.75,
            0.75,
            ax1,
            color=color_goal,
        )

        leg1 = ax1.legend(
            loc="upper center",
            ncol=2,
            fontsize=legend_sz,
            bbox_to_anchor=(0.5, 1.015),
            borderpad=0.27,
            columnspacing=0.6,
            handletextpad=0.05,
            handlelength=1.2,
            fancybox=True,
            shadow=True,
        )
        leg1.get_title().set_multialignment("center")
        plt.tight_layout()

        border_lw = 1.5

        for spine in ax1.spines.values():
            spine.set_linewidth(border_lw)

        ax1.text(
            0,
            3,
            "unsafe",
            fontsize=ticks_sz,
            color=obstacle_color,
            ha="center",
            va="center",
        )

        # TODO Move location of this later
        ax1.text(
            3,
            3,
            r"$\mathcal{C}_{\rm S}$",
            fontsize=ticks_sz,
            color="black",
            ha="center",
            va="center",
        )

        # if env.vanilla_flag:
        #     label = "(a)"
        # elif env.gen_flag:
        #     label = "(b)"
        # else:
        #     label = "(c)"

        # ax1.text(
        #     -3.6,
        #     -0.5,
        #     label,
        #     fontsize=ticks_sz,
        #     color="black",
        # )

        if show_plots:
            plt.show()

        if save_plots:
            plt.savefig(
                "plots/quadrotor_xz_" + mode + ".pdf",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0.02,
            )

        if u_plots:
            # Legend stuff
            frame_bool = True
            shadow_bool = False
            fancy_bool = True

            ##################### CONTROL PLOT #####################
            # ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
            plt.figure(figsize=(2 * n, 0.9 * n), dpi=dpi)

            ##################### 2D PLOT #####################
            # ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=2)
            ax3 = plt.gca()
            ax3.yaxis.tick_right()
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax3.set_xlim([0, t_span[-1]])
            ax3.set_ylim([-20, 20])  # hardcoded
            for i in range(2):
                ax3.grid(False)
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                ax3.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits

                if i == 0:
                    color = "#231942"
                    label = "$F"
                elif i == 1:
                    color = "#9F86C0"
                    label = "$M"

                ax3.plot(
                    t_span_u,
                    u_p[i, 1:],
                    "--",
                    color=color,
                    label=label + "_{\\rm p}$",
                    linewidth=lwp,
                )
                ax3.plot(
                    t_span_u,
                    u_act[i, 1:],
                    "-",
                    color=color,
                    label=label + "_{\\rm safe}$",
                    linewidth=lwp,
                )

            ax3.legend(
                fontsize=legend_sz,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.35),
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=shadow_bool,
                ncol=4,
            )
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            plt.tight_layout()

            if show_plots:
                plt.show()

            if save_plots:
                plt.savefig(
                    "plots/quadrotor_u_" + mode + ".pdf",
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )