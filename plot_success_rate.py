import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define Colors
COLORS = {
    "biomorphic_blue": "#0066A2",
    "biomorphic_blue_complimentary": "#FE8C00",
    "delft_blue": "#00A6D6",
    "color_x": "#F80031",
    "color_y": "#FFC700",
    "color_z": "#FF8100",
    "dark_grey": "#2e2e2e",
    "color_contact": "red"
}

# Matplotlib Configuration
plt.rcParams['text.usetex'] = True
plt.rc('font', family='normal', weight='bold', size=31)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plotting script for Evaluating Monte Carlo Sim")
    parser.add_argument('--path', type=str, required=True, help="Path to data files")
    parser.add_argument('--pos_range', type=str, required=True, help="Space-separated list of \"min max step\"")
    parser.add_argument('--ang_range', type=str, required=True, help="Space-separated list of \"min max step\"")
    return parser.parse_args()

def load_data(file_path):
    data = np.load(file_path)
    return data['state_machine_states'], data['t']

def compute_metrics(value_range, data_path_template):
    num_values = len(value_range)
    success_rates = np.zeros(num_values)
    time_to_perch = np.zeros(num_values)
    time_to_perch_std = np.zeros(num_values)
    
    for i, value in enumerate(value_range):
        file_path = data_path_template.format(value)
        states, t = load_data(file_path)
        
        perch_indices = np.argmax(states == 5, axis=1)
        perch_indices[~np.any(states == 5, axis=1)] = -1
        
        time_to_perch[i] = np.mean(t[perch_indices])
        time_to_perch_std[i] = np.std(t[perch_indices])
        success_rates[i] = np.sum(states[:, -1] == 5) / states.shape[0] * 100.0
    
    return success_rates, time_to_perch, time_to_perch_std

def plot_results(pos_ran, ang_ran, success_rate_pos, success_rate_ang, time_to_perch_pos, time_to_perch_ang, time_to_perch_std_pos, time_to_perch_std_ang):
    fig, axs = plt.subplots(2, figsize=(25, 13))
    fig.subplots_adjust(bottom=-0.1)
    axs_right = [ax.twinx() for ax in axs]

    # Success Rate Bar Plots
    axs[0].bar(pos_ran, success_rate_pos, color=COLORS["biomorphic_blue"], width=0.9 * (pos_ran[1] - pos_ran[0]), alpha=0.8)
    axs[1].bar(ang_ran, success_rate_ang, color=COLORS["biomorphic_blue"], width=0.9 * (ang_ran[1] - ang_ran[0]), alpha=0.8)

    # Formatting
    axs[0].set_xlabel(r"Position Offset [$m$]")
    axs[0].set_ylabel(r"Success Rate [\%]", color=COLORS["biomorphic_blue"])
    axs_right[0].set_ylabel(r"Time-to-Perch [$s$]")
    axs[0].set_ylim([0, 100])

    axs[1].set_xlabel(r"Angular Offset [$^\circ$]")
    axs[1].set_ylabel(r"Success Rate [\%]", color=COLORS["biomorphic_blue"])
    axs_right[1].set_ylabel(r"Time-to-Perch [$s$]")
    axs[1].set_xticks(np.linspace(-90, 90, 5))

    # Set xlims
    axs[0].set_xlim([pos_ran[0], pos_ran[-1]])
    axs[1].set_xlim([ang_ran[0], ang_ran[-1]])
    axs[1].set_ylim([0, 100])
    
    # Time-to-Perch Error Bars
    axs_right[0].errorbar(pos_ran, time_to_perch_pos, time_to_perch_std_pos, linestyle='None', marker='o', color='black', capsize=5, markersize=5)
    axs_right[1].errorbar(ang_ran, time_to_perch_ang, time_to_perch_std_ang, linestyle='None', marker='o', color='black', capsize=5, markersize=5)
    
    fig.savefig("perching_success.svg", bbox_inches="tight")

def main():
    args = parse_arguments()
    
    pos_ran = np.arange(*np.fromstring(args.pos_range, sep=" "))
    ang_ran = np.arange(*np.fromstring(args.ang_range, sep=" "), dtype=int)
    
    success_rate_pos, time_to_perch_pos, time_to_perch_std_pos = compute_metrics(pos_ran, args.path + "position/trial_{:.2f}.npz")
    success_rate_ang, time_to_perch_ang, time_to_perch_std_ang = compute_metrics(ang_ran, args.path + "angle/trial_{:02d}.npz")
    
    plot_results(pos_ran, ang_ran, success_rate_pos, success_rate_ang, time_to_perch_pos, time_to_perch_ang, time_to_perch_std_pos, time_to_perch_std_ang)

if __name__ == "__main__":
    main()
