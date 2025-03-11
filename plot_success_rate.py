import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_po():
    parser = argparse.ArgumentParser(description="Plotting script for Evaluating Monte Carlo Sim")
    parser.add_argument('--path', type=str, help="Path to data files")
    parser.add_argument('--float', action='store_true', help="Whether the range is defined using float values")
    parser.add_argument('--range', type=str, help="Space-separated list of \"min max step\"")
    return parser.parse_args()

def main():
    args = read_po()
    ran = np.arange(*np.fromstring(args.range, sep=" "))
    success_rate = np.zeros_like(ran)

    for i, val in enumerate(ran):
        if args.float:
            data = np.load(args.path + f"trial_{val:.1f}.npz")
        else:
            data = np.load(args.path + f"trial_{int(val):02}.npz")

        states = data['state_machine_states']
        success = states[:, -1] == 5
        success_rate[i] = np.sum(success, axis=0)[0]

    plt.scatter(ran, success_rate)
    plt.show()      

if __name__=="__main__":
    main()