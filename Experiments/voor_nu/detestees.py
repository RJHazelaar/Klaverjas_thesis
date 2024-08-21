import numpy as np
import time
import os
import math
import sys
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.realpath(os.path.join(__file__ ,"../..")))
sys.path.append(parent_dir)
scores_dir = parent_dir + "/Experiments/voor_nu"

def run_test():
    az3 = np.loadtxt(scores_dir+"/run_experiment_alphazero_3.txt")
    az4 = np.loadtxt(scores_dir+"/run_experiment_alphazero_4.txt")
    pimc3_40_200 = np.loadtxt(scores_dir+"/run_experiment_pimc_3_40_200.txt")
    pimc3_40_800 = np.loadtxt(scores_dir+"/run_experiment_pimc_3_40_800.txt")
    pimc5_200_200 = np.loadtxt(scores_dir+"/run_experiment_pimc_5_200_200.txt")
    pimc5_200_800 = np.loadtxt(scores_dir+"/run_experiment_pimc_5_200_800.txt")

    np.set_printoptions(precision=3)

    az3_wins = 0
    rule_wins = 0
    for score in az3:
        if score > 0:
            az3_wins += 1
        else:
            rule_wins += 1
    winrate = az3_wins/(az3_wins +rule_wins)
    print("winrate az3")
    print(winrate)
    print(np.mean(az3))

    pimc3_40_200_wins = 0
    rule_wins = 0
    for score in pimc3_40_200:
        if score > 0:
            pimc3_40_200_wins += 1
        else:
            rule_wins += 1
    winrate = pimc3_40_200_wins/(pimc3_40_200_wins +rule_wins)
    print("winrate pimc3_40_200")
    print(winrate)
    print(np.mean(pimc3_40_200))

    pimc5_200_200_wins = 0
    rule_wins = 0
    for score in pimc5_200_200:
        if score > 0:
            pimc5_200_200_wins += 1
        else:
            rule_wins += 1
    winrate = pimc5_200_200_wins/(pimc5_200_200_wins +rule_wins)
    print("winrate pimc5_200_200")
    print(winrate)
    print(np.mean(pimc5_200_200))

    ######################################################################################
    
    az4_wins = 0
    rule_wins = 0
    for score in az4:
        if score > 0:
            az4_wins += 1
        else:
            rule_wins += 1
    winrate = az4_wins/(az4_wins +rule_wins)
    print("winrate az4")
    print(winrate)
    print(np.mean(az4))

    pimc3_40_800_wins = 0
    rule_wins = 0
    for score in pimc3_40_800:
        if score > 0:
            pimc3_40_800_wins += 1
        else:
            rule_wins += 1
    winrate = pimc3_40_800_wins/(pimc3_40_800_wins +rule_wins)
    print("winrate pimc3_40_800")
    print(winrate)
    print(np.mean(pimc3_40_800))

    pimc5_200_800_wins = 0
    rule_wins = 0
    for score in pimc5_200_800:
        if score > 0:
            pimc5_200_800_wins += 1
        else:
            rule_wins += 1
    winrate = pimc5_200_800_wins/(pimc5_200_800_wins +rule_wins)
    print("winrate pimc5_200_800")
    print(winrate)
    print(np.mean(pimc5_200_800))

    x = range(len(pimc5_200_800))
    y = pimc5_200_800
    plt.scatter(x, y, s = 1.1 ,alpha=0.5, color="purple")
    #x2 = range(len(pimc3_40_800))
    #y2 = pimc3_40_800
    #plt.scatter(x2, y2, s = 0.7 ,alpha=0.5, color="green")
    x3 = range(len(az4))
    y3 = az4
    plt.scatter(x3, y3, s = 1.1 ,alpha=0.5, color="blue")
    plt.show()


    az4 = az4[: len(az4) - 20]
    print(len(az4))
    print(len(pimc3_40_800))
    print(len(pimc5_200_800))

    x = np.vstack([az4, pimc3_40_800, pimc5_200_800])
    x = x.transpose()

    n_bins = 30
    colors = ['red', 'orange', 'blue']
    labels = ['ISMCTS', 'PIMC-40', 'PIMC-200']
    plt.hist(x, bins=n_bins, density=True, histtype='bar', color=colors, label=labels)
    plt.legend(prop={'size': 10})
    plt.show()


if __name__ == "__main__":
    run_test()

