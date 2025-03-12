import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

# Define Colors & Linestyles
biomorphic_blue = "#0066A2"
biomorphic_blue_complimentary = "#FE8C00"
delft_blue = "#00A6D6"
delft_blue_darker = "#258EFC"
delft_blue_ddarker = "#002FDC"
color_x = "#F80031"
color_y = "#FFC700"
color_z = "#FF8100"
bright_grey = "#a1a095ff"
dark_grey = "#2e2e2e"
color_contact="red"
linestyle0 = "-"
linestyle1 = "--"
linestyle2 = ":"
colors = [color_x, color_y, delft_blue]

# Enable LaTeX text rendering
plt.rcParams['text.usetex'] = True
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 31}

matplotlib.rc('font', **font)

def read_po():
    parser = argparse.ArgumentParser(description="Plotting script for Evaluating Monte Carlo Sim")
    parser.add_argument('--path', type=str, help="Path to data files")
    parser.add_argument('--pos_range', type=str, help="Space-separated list of \"min max step\"")
    parser.add_argument('--ang_range', type=str, help="Space-separated list of \"min max step\"")
    return parser.parse_args()

def main():
    args = read_po()
    pos_ran = np.arange(*np.fromstring(args.pos_range, sep=" "))
    ang_ran = np.arange(*np.fromstring(args.ang_range, sep=" "))
    success_rate_pos = np.zeros_like(pos_ran)
    success_rate_ang = np.zeros_like(ang_ran)    
    time_to_perch_pos = np.zeros_like(pos_ran)   
    time_to_perch_ang = np.zeros_like(ang_ran)
    time_to_perch_std_pos = np.zeros_like(pos_ran)
    time_to_perch_std_ang = np.zeros_like(ang_ran)

    for i, val in enumerate(pos_ran):
        data_pos = np.load(args.path + "position/" + f"trial_{val:.2f}.npz")

        states = data_pos['state_machine_states']
        t = data_pos['t']

        # Find the first occurrence of p in each row
        time_to_perch_idx = np.argmax(states == 5, axis=1)

        # Handle cases where p is not found in a row
        time_to_perch_idx[~np.any(states == 5, axis=1)] = -1  

        # Compute the mean time to perch
        time_to_perch_pos[i] = np.mean(t[time_to_perch_idx])
        time_to_perch_std_pos[i] = np.std(t[time_to_perch_idx])
        
        # Define success rate
        success = states[:, -1] == 5
        success_rate_pos[i] = np.sum(success, axis=0)[0] / states.shape[0] * 100.0

    for i, val in enumerate(ang_ran):
        data_ang = np.load(args.path + "angle/" + f"trial_{int(val):02d}.npz")

        states = data_ang['state_machine_states']
        t = data_ang['t']

        # Find the first occurrence of p in each row
        time_to_perch_idx = np.argmax(states == 5, axis=1)

        # Handle cases where p is not found in a row
        time_to_perch_idx[~np.any(states == 5, axis=1)] = -1  

        # Compute the mean time to perch
        time_to_perch_ang[i] = np.mean(t[time_to_perch_idx])
        time_to_perch_std_ang[i] = np.std(t[time_to_perch_idx])
        
        # Define success rate
        success = states[:, -1] == 5
        success_rate_ang[i] = np.sum(success, axis=0)[0] / states.shape[0] * 100.0

    
    fig, axs = plt.subplots(2)
    fig.subplots_adjust(bottom=0.0)
    handles = []
    axs_right = [axs[0].twinx(), axs[1].twinx()]

    #max_idx = np.argmax(success_rate)
    #axs[0].text(ran[max_idx]-0.35*(ran[1] - ran[0]),
    #            success_rate[max_idx] - 10,
    #            f"{int(success_rate[max_idx]):01d}\\%", size=25)
    handles.append(axs[0].bar(pos_ran, success_rate_pos, color=biomorphic_blue,
            width=0.9 * (pos_ran[1]-pos_ran[0]), alpha=0.5, zorder=1))
    handles.append(
        axs[1].bar(ang_ran, success_rate_ang, color=biomorphic_blue_complimentary,
               width=0.9 * (ang_ran[1]-ang_ran[0]), alpha=0.5, zorder=1))
    axs[0].set_xlabel(r"Position Offset [$ m $]", labelpad=7)
    axs[0].set_ylabel(r"Success Rate [\%]", labelpad=7)
    axs_right[0].set_ylabel(r"Time-to-Perch [$ s $]", labelpad=7)
    axs[0].set_ylim([0, 100])
    axs[0].set_xlim([pos_ran[0] - 0.5*(pos_ran[1]-pos_ran[0]),
                     pos_ran[-1] + 0.5*(pos_ran[1]-pos_ran[0])])

    handles.append(
        axs_right[0].scatter(pos_ran, time_to_perch_pos, color=biomorphic_blue,
                   marker='o', s=100, label=r"$t$", zorder=10)
    )    
    handles.append(
        axs_right[1].scatter(ang_ran, time_to_perch_ang, color=biomorphic_blue_complimentary,
                   marker='o', s=100, label=r"$t$", zorder=10)
    )
    handles.append(
        axs_right[0].errorbar(pos_ran, time_to_perch_pos, time_to_perch_std_pos,
                    linestyle='None', marker='None', color=bright_grey,
                    capsize=5, label=r"$\sigma_t$", zorder=10)
    )
    handles.append(
        axs_right[1].errorbar(ang_ran, time_to_perch_ang, time_to_perch_std_ang,
                    linestyle='None', marker='None', color=bright_grey,
                    capsize=5, label=r"$\sigma_t$", zorder=10)
    )
    axs[1].set_ylabel(r"Success Rate [\%]", labelpad=7)
    axs_right[1].set_ylabel(r"Time-to-Perch [$ s $]", labelpad=7)
    axs[1].set_xlabel(r"Angular Offset [$^\circ$]", labelpad=7)
    axs[1].set_xticks(np.linspace(-90, 90, 5))
    axs[1].set_xlim([ang_ran[0] - 0.5*(ang_ran[1]-ang_ran[0]),
                     ang_ran[-1] + 0.5*(ang_ran[1]-ang_ran[0])])
    axs_right[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.21), ncols=len(handles), framealpha=0.5,
                  handleheight=0.3,
                  labelspacing=(17.4 - len(handles) * 0.3) / (len(handles)-1),
                  handlelength=1.0,
                  handletextpad=0.25,
                  handles=handles, labels=[r"$\mathcal{R}_{\mathrm{Pos}}$",
                                           r"$\mathcal{R}_{\mathrm{Ang}}$",
                                           r"$\mathcal{T}_{\mathrm{Pos}}$",
                                           r"$\mathcal{T}_{\mathrm{Ang}}$",
                                           r"$\sigma_{\mathcal{T}_\mathrm{Pos}}$",
                                           r"$\sigma_{\mathcal{T}_\mathrm{Ang}}$"])

    fig.set_size_inches([21, 13])
    fig.savefig("perching_success.svg", bbox_inches="tight")    

if __name__=="__main__":
    main()