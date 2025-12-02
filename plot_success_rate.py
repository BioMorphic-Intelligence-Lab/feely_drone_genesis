import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from feely_drone_common.state_machine import State

# Define Colors
COLORS = {
    "biomorphic_blue": "#0066A2",
    "biomorphic_blue_grayed": "#5B88A1",
    "biomorphic_blue_complimentary": "#FE8C00",
    "biomorphic_blue_complimentary_grayed": "#FFCA89",
    "delft_blue": "#00A6D6",
    "color_x": "#F80031",
    "color_y": "#FFC700",
    "color_z": "#FF8100",
    "dark_grey": "#2e2e2e",
    "color_contact": "red"
}

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plotting script for Evaluating Monte Carlo Sim")
    parser.add_argument('--path', type=str, required=True, help="Path to data files")
    parser.add_argument('--pos_range', type=str, required=True, help="Space-separated list of position offset \"min max step\"")
    parser.add_argument('--ang_range', type=str, required=True, help="Space-separated list of angular offset \"min max step\"")
    parser.add_argument('--rad_range', type=str, required=True, help="Space-separated list of cylinder radius \"min max step\"")
    return parser.parse_args()

def load_data(file_path):
    data = {}
    try:
        data = np.load(file_path)
    except:
        data['state_machine_states'] = np.zeros([1, 1])
        data['positions'] = np.zeros([1, 1, 3])
        data['t'] = 60.0 * np.ones(1)
    return data['state_machine_states'], data['positions'], data['t']

def compute_metrics(value_range, data_path_template):
    num_values = len(value_range)
    success_rates = np.zeros(num_values)
    time_to_perch = np.zeros(num_values)
    time_to_perch_std = np.zeros(num_values)
    
    for i, value in enumerate(value_range):
        file_path = data_path_template.format(value)
        states, positions, t = load_data(file_path)
        
        perch_indices = np.argmax(states == State.PERCH.value, axis=1).flatten()
        perch_indices[~np.any(states == State.PERCH.value, axis=1).flatten()] = -1

        end_height =  (positions[:, -1, 2] > 1.75).flatten()
        perched = (states[:, -1] == State.PERCH.value).flatten()
        # Make sure we're actually perched by checking the height
        # otherwise set perch index to -1
        perch_indices[~end_height] = -1
            
        time_to_perch[i] = np.mean(t[perch_indices])
        time_to_perch_std[i] = np.std(t[perch_indices])
        success_rates[i] = np.sum(
            np.logical_and(
                perched, end_height
            )
        ) / states.shape[0] * 100.0
    
    return success_rates, time_to_perch, time_to_perch_std

def init_plot():

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3, hspace=0.075, wspace=0.025,
                           left=0.08, right=0.98, top=0.90, bottom=0.2) 
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    axs = [ax1, ax2, ax3]
    fig.set_size_inches(np.array([2.0, 1.0]) * 10)

    #fig.subplots_adjust(bottom=-0.1)
    ax1_bottom = fig.add_subplot(gs[1, 0])
    ax2_bottom = fig.add_subplot(gs[1, 1], sharey=ax1_bottom)
    ax3_bottom = fig.add_subplot(gs[1, 2], sharey=ax1_bottom)
    axs_bottom = [ax1_bottom, ax2_bottom, ax3_bottom]

    # Formatting
    axs_bottom[0].set_xlabel(r"Position Offset [\$m\$]")
    axs[0].set_ylabel(r"Success Rate [%]")
    axs_bottom[0].set_ylabel(r"Time-to-Perch [\$s\$]")
    axs_bottom[0].set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    axs[0].set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    axs[0].set_yticks(np.linspace(0, 100, 5, endpoint=True))
    axs_bottom[0].set_yticks(np.linspace(0, 60, 4, endpoint=True))

    axs_bottom[1].set_xlabel(r"Angular Offset [\$^\circ\$]")
    axs_bottom[1].set_xticks(np.linspace(-90, 90, 5))
    axs[1].set_xticks(np.linspace(-90, 90, 5))
    axs[1].set_yticks(np.linspace(0, 100, 5, endpoint=True))
    axs_bottom[1].set_yticks(np.linspace(0, 60, 4, endpoint=True))

    axs_bottom[2].set_xlabel(r"Cylinder Radius [\$m\$]")
    axs_bottom[2].set_xticks(np.linspace(0.0, 0.3, 4, endpoint=True))
    axs[2].set_xticks(np.linspace(0.0, 0.3, 4, endpoint=True))
    axs[2].set_yticks(np.linspace(0, 100, 5, endpoint=True))
    axs_bottom[2].set_yticks(np.linspace(0, 60, 4, endpoint=True))

    axs_bottom[0].set_ylim([0, 62])
    axs_bottom[1].set_ylim([0, 62])
    axs_bottom[2].set_ylim([0, 62])

    xlabelpad = 20
    ylabelpad = 45
    tickpad = 20
    axs_bottom[0].xaxis.labelpad = xlabelpad
    axs_bottom[0].tick_params(axis='both', pad=tickpad)
    axs_bottom[1].xaxis.labelpad = xlabelpad
    axs_bottom[1].tick_params(axis='both', pad=tickpad)
    axs_bottom[1].tick_params(axis='y', labelleft=False)
    axs_bottom[2].xaxis.labelpad = xlabelpad
    axs_bottom[2].tick_params(axis='both', pad=tickpad)
    axs_bottom[2].tick_params(axis='y', labelleft=False)

    axs[0].xaxis.labelpad = xlabelpad
    axs[0].tick_params(axis='both', pad=tickpad)
    axs[0].tick_params(axis='y', labelright=False)
    axs[0].set_xticklabels([])
    axs[1].tick_params(axis='y', labelright=False)
    axs[1].tick_params(axis='y', labelleft=False)
    axs[1].set_xticklabels([])
    axs[1].xaxis.labelpad = xlabelpad
    axs[2].xaxis.labelpad = xlabelpad
    axs[2].set_xticklabels([])
    axs[2].tick_params(axis='both', pad=tickpad)
    axs[2].tick_params(axis='y', labelleft=False)

    axs[0].yaxis.labelpad = ylabelpad
    axs[1].yaxis.labelpad = ylabelpad
    axs[2].yaxis.labelpad = ylabelpad
    axs_bottom[0].yaxis.labelpad = ylabelpad
    axs_bottom[1].yaxis.labelpad = ylabelpad
    axs_bottom[2].yaxis.labelpad = ylabelpad

    return fig, axs, axs_bottom

def plot_results(fig, axs, axs_bottom, color, label,
                 pos_ran, ang_ran, rad_ran,
                 success_rate_pos, success_rate_ang, success_rate_rad,
                 time_to_perch_pos, time_to_perch_ang, time_to_perch_rad,
                 time_to_perch_std_pos, time_to_perch_std_ang, time_to_perch_std_rad):
        
    # Success Rate Bar Plots
    axs[0].step(pos_ran, success_rate_pos, color=color, label=label, where='post')
    #fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    #, width=0.9 * (pos_ran[1] - pos_ran[0]), alpha=0.8)
    axs[1].step(ang_ran, success_rate_ang, color=color, where='post')#, width=0.9 * (ang_ran[1] - ang_ran[0]), alpha=0.8)
    axs[2].step(rad_ran, success_rate_rad, color=color, where='post')#, width=0.9 * (rad_ran[1] - rad_ran[0]), alpha=0.8)

    # Set xlims
    axs[0].set_xlim([pos_ran[0]  - 0.5 * (pos_ran[1] - pos_ran[0]),
                     pos_ran[-1] + 0.5 * (pos_ran[1] - pos_ran[0])])
    axs[1].set_xlim([ang_ran[0]  - 0.5 * (ang_ran[1] - ang_ran[0]),
                     ang_ran[-1] + 0.5 * (ang_ran[1] - ang_ran[0])])
    axs[2].set_xlim([0,
                     rad_ran[-1] + 0.5 * (rad_ran[1] - rad_ran[0])])
    axs_bottom[0].set_xlim([pos_ran[0]  - 0.5 * (pos_ran[1] - pos_ran[0]),
                            pos_ran[-1] + 0.5 * (pos_ran[1] - pos_ran[0])])
    axs_bottom[1].set_xlim([ang_ran[0]  - 0.5 * (ang_ran[1] - ang_ran[0]),
                            ang_ran[-1] + 0.5 * (ang_ran[1] - ang_ran[0])])
    axs_bottom[2].set_xlim([0,
                            rad_ran[-1] + 0.5 * (rad_ran[1] - rad_ran[0])])
    axs[0].set_ylim([0, 102.5])
    axs[1].set_ylim([0, 102.5])
    axs[2].set_ylim([0, 102.5])
    
    # Time-to-Perch Error Bars
    axs_bottom[0].errorbar(pos_ran, time_to_perch_pos, time_to_perch_std_pos, linestyle='None', marker='o', color=color, capsize=2, markersize=0, alpha=0.3)
    axs_bottom[0].step(pos_ran, time_to_perch_pos, color=color, zorder=10, where='post')
    axs_bottom[1].errorbar(ang_ran, time_to_perch_ang, time_to_perch_std_ang, linestyle='None', marker='o', color=color, capsize=2, markersize=0, alpha=0.3)
    axs_bottom[1].step(ang_ran, time_to_perch_ang, color=color, zorder=10, where='post')
    axs_bottom[2].errorbar(rad_ran, time_to_perch_rad, time_to_perch_std_rad, linestyle='None', marker='o', color=color, capsize=2, markersize=0, alpha=0.3)
    axs_bottom[2].step(rad_ran, time_to_perch_rad, color=color, zorder=10, where='post')

def main():
    args = parse_arguments()
    
    pos_ran = np.arange(*np.fromstring(args.pos_range, sep=" "))
    ang_ran = np.arange(*np.fromstring(args.ang_range, sep=" "), dtype=int)
    rad_ran = np.arange(*np.fromstring(args.rad_range, sep=" "))
    
    fig, axs, axs_bottom = init_plot()

    success_rate_pos, time_to_perch_pos, time_to_perch_std_pos = compute_metrics(pos_ran, args.path + "logs/position/trial_{:.2f}.npz")
    success_rate_ang, time_to_perch_ang, time_to_perch_std_ang = compute_metrics(ang_ran, args.path + "logs/angle/trial_{:02d}.npz")
    success_rate_rad, time_to_perch_rad, time_to_perch_std_rad = compute_metrics(rad_ran, args.path + "logs/radius/trial_{:.3f}.npz")

    success_rate_pos_baseline, time_to_perch_pos_baseline, time_to_perch_std_pos_baseline = compute_metrics(pos_ran, args.path + "logs_simple/position/trial_{:.2f}.npz")
    success_rate_ang_baseline, time_to_perch_ang_baseline, time_to_perch_std_ang_baseline = compute_metrics(ang_ran, args.path + "logs_simple/angle/trial_{:02d}.npz")
    success_rate_rad_baseline, time_to_perch_rad_baseline, time_to_perch_std_rad_baseline = compute_metrics(rad_ran, args.path + "logs_simple/radius/trial_{:.3f}.npz")
    
    plot_results(fig, axs, axs_bottom, COLORS["biomorphic_blue_complimentary"],
                 "Non-tactile Feedforward Perching",
                 pos_ran, ang_ran, rad_ran,
                 success_rate_pos_baseline, success_rate_ang_baseline, success_rate_rad_baseline,
                 time_to_perch_pos_baseline, time_to_perch_ang_baseline, time_to_perch_rad_baseline,
                 time_to_perch_std_pos_baseline, time_to_perch_std_ang_baseline, time_to_perch_std_rad_baseline)
    
    plot_results(fig, axs, axs_bottom, COLORS["biomorphic_blue"],
                 "Tactile Perching (\\textbf{Ours})                                                                                             ",
                 pos_ran, ang_ran, rad_ran,
                 success_rate_pos, success_rate_ang, success_rate_rad,
                 time_to_perch_pos, time_to_perch_ang, time_to_perch_rad,
                 time_to_perch_std_pos, time_to_perch_std_ang, time_to_perch_std_rad)
    

    # Create an *empty global legend* on top of figure
    # bbox_to_anchor moves the legend above the top of the axes
    legend = fig.legend(loc="lower left", 
                        bbox_to_anchor=(0.075, 0.92, 1.4, 0.),
                        ncol=2, frameon=True,
                        handlelength=5.0,   # length of line in legend
                        columnspacing=51,
                        borderpad=2.5)

    fig.savefig("perching_success.svg", bbox_inches='tight')

if __name__ == "__main__":
    main()
