import matplotlib.pyplot as plt
import numpy as np


# results {'mcts_steps': 10, 'n_of_sims': 0, 'nn_scaler': 1, 'ucb_c': 50}
# time: 427.05185437202454
# score: 14.3 std_score: 1.3 eval_time(ms): 24.2
# results {'mcts_steps': 50, 'n_of_sims': 0, 'nn_scaler': 1, 'ucb_c': 200}
# time: 1680.2017540931702
# score: 30.6 std_score: 1.3 eval_time(ms): 102

c_values = [3, 10, 50, 200, 800, 3200, 12800, 51200, 204800]

X_10_1 = [3, 10, 50, 200, 800]
y_10_1 = [0.9, 3.0, 4.0, 3.0, -6.9]
std_10_1 = 1.3

X_50_1 = [10, 50, 200, 800, 3200]
y_50_1 = [17.3, 22.9, 28.8, 29.1, 16.4]
std_50_1 = 1.3

X_200_1 = [50, 200, 800, 3200, 12800]
y_200_1 = [35.3, 45.6, 43.9, 43.7, 30.7]
std_200_1 = 1.3

X_800_1 = [50, 200, 800, 3200, 12800, 51200]
y_800_1 = [43.9, 54.9, 53.8, 55.0, 54.5, 46.1]
std_800_1 = 1.8

X_3200_1 = [50, 200, 800, 3200, 12800, 51200, 204800]
y_3200_1 = [50.4, 61.5, 60.7, 59, 58.6, 57.7, 53.6]
std_3200_1 = 2.3

marker = ".-"
alpha1 = 0.08
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig2, ax2 = plt.subplots(figsize=(7, 7))
fig3, ax3 = plt.subplots(figsize=(7, 7))
fig4, ax4 = plt.subplots(figsize=(7, 7))

ax1.plot(X_10_1, y_10_1, marker, label="10 simulations")
ax1.fill_between(X_10_1, np.array(y_10_1) - std_10_1 * 1.96, np.array(y_10_1) + std_10_1 * 1.96, alpha=alpha1)

ax1.plot(X_50_1, y_50_1, marker, label="50 simulations")
ax1.fill_between(X_50_1, np.array(y_50_1) - std_50_1 * 1.96, np.array(y_50_1) + std_50_1 * 1.96, alpha=alpha1)

ax1.plot(X_200_1, y_200_1, marker, label="200 simulations")
ax1.fill_between(X_200_1, np.array(y_200_1) - std_200_1 * 1.96, np.array(y_200_1) + std_200_1 * 1.96, alpha=alpha1)

ax1.plot(X_800_1, y_800_1, marker, label="800 simulations")
ax1.fill_between(X_800_1, np.array(y_800_1) - std_800_1 * 1.96, np.array(y_800_1) + std_800_1 * 1.96, alpha=alpha1)

ax1.plot(X_3200_1, y_3200_1, marker, label="3200 simulations")
ax1.fill_between(
    X_3200_1, np.array(y_3200_1) - std_3200_1 * 1.96, np.array(y_3200_1) + std_3200_1 * 1.96, alpha=alpha1
)

ax1.vlines(
    c_values,
    -10,
    [0.9, 17.3, 50.4, 61.5, 60.7, 59, 58.6, 57.7, 53.6],
    linestyles="dashed",
    color="black",
    alpha=0.2,
    linewidth=1,
)
ax1.set_ylim(bottom=-10)

ax1.set_xscale("log")
ax1.set_xlabel("Exploration Rate (C value)")
ax1.set_ylabel("Score Difference")
# ax1.set_title("Effect of different C values on \nthe average score (rollouts=1).")
ax1.legend()


steps = [10, 50, 200, 800, 3200]

time_hp = [24.2, 102, 410, 1708, 6631.9]
score_hp = [14.3, 30.6, 42.7, 50.5, 53.9]
std_hp = np.array([1.3, 1.3, 1.3, 2.3, 4])

time_sp = [32.8, 229.8, 395.2, 1551.9, 6561.3]
score_sp = [8.9, 20.1, 34.8, 45.8, 46.4]
std_sp = np.array([1.3, 1.3, 1.8, 2.3, 4.0])

time_rollout = [2.9, 12.6, 54.1, 310, 885.2]
score_rollout = [4, 28.8, 43.9, 55, 58.6]
std_rollout = np.array([1.3, 1.3, 1.3, 1.8, 2.3])

ax2.plot(steps, score_hp, marker, label="Human-data")
ax2.fill_between(
    steps,
    np.array(score_hp) - std_hp * 1.96,
    np.array(score_hp) + std_hp * 1.96,
    alpha=alpha1,
    edgecolor="blue",
)
ax2.plot(steps, score_sp, marker, label="Self-play")
ax2.fill_between(
    steps,
    np.array(score_sp) - std_sp * 1.96,
    np.array(score_sp) + std_sp * 1.96,
    alpha=alpha1,
    edgecolor="orange",
)
ax2.plot(steps, score_rollout, marker, label="Rollout")
ax2.fill_between(
    steps,
    np.array(score_rollout) - std_rollout * 1.96,
    np.array(score_rollout) + std_rollout * 1.96,
    alpha=alpha1,
    edgecolor="green",
)
all_y = score_hp + score_sp + score_rollout
ax2.vlines(steps, 0, [14.3, 30.6, 43.9, 55, 58.6], linestyles="dashed", color="black", alpha=0.2, linewidth=1)
ax2.set_ylim(bottom=0)
ax2.set_xscale("log")
# ax2.set_title("Effect of different number of simulations \non the average score (rollouts=1).")
ax2.set_xlabel("Tree Depth per move (simulations)")
ax2.set_ylabel("Score Difference")
ax2.legend()

number_base = 130
number_scale = 10

ax3.plot(time_hp, score_hp, marker, label="Simulations (Human-data)")
ax3.fill_between(
    time_hp,
    np.array(score_hp) - std_hp * 1.96,
    np.array(score_hp) + std_hp * 1.96,
    alpha=alpha1,
    edgecolor="blue",
)
for xp, yp, m in zip(time_hp, score_hp, steps):
    ax3.scatter(
        xp,
        yp,
        marker=f"${m}$",
        s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
        color="black",
        zorder=10,
    )
ax3.plot(time_sp, score_sp, marker, label="Simulations (Self-play)")
ax3.fill_between(
    time_sp,
    np.array(score_sp) - std_sp * 1.96,
    np.array(score_sp) + std_sp * 1.96,
    alpha=alpha1,
    edgecolor="orange",
)
score_sp[0] += 1.5
for xp, yp, m in zip(time_sp, score_sp, steps):
    ax3.scatter(
        xp,
        yp,
        marker=f"${m}$",
        s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
        color="black",
        zorder=10,
    )
ax3.plot(time_rollout, score_rollout, marker, label="Simulations (Rollout)")
ax3.fill_between(
    time_rollout,
    np.array(score_rollout) - std_rollout * 1.96,
    np.array(score_rollout) + std_rollout * 1.96,
    alpha=alpha1,
    edgecolor="green",
)
score_rollout[0] += 2
for xp, yp, m in zip(time_rollout, score_rollout, steps):
    ax3.scatter(
        xp,
        yp,
        marker=f"${m}$",
        s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
        color="black",
        zorder=10,
    )
sims_step10_c1 = [200, 50, 15, 3, 1]
time_step10_c1 = [345, 91, 30.4, 8.48, 2.9]
score_step10_c1 = [12.8, 10.3, 7.8, 4.8, 4]
std_step10_c1 = 0.6

ax3.plot(time_step10_c1, score_step10_c1, marker, label="Random Rollouts")
ax3.fill_between(
    time_step10_c1,
    np.array(score_step10_c1) - std_step10_c1 * 1.96,
    np.array(score_step10_c1) + std_step10_c1 * 1.96,
    alpha=alpha1,
    edgecolor="red",
)
time_step10_c1[4] += 1
score_step10_c1[2] -= 0.5
for xp, yp, m in zip(time_step10_c1, score_step10_c1, sims_step10_c1):
    ax3.scatter(
        xp,
        yp,
        marker=f"${m}$",
        s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
        color="black",
        zorder=10,
    )

# ax3.set_title("Effect of different number of simulations and rollouts on\n the think time and average score.")
ax3.set_xscale("log")
ax3.set_xlabel("Thinking time per move (ms)")
ax3.set_ylabel("Score Difference")
ax3.legend(loc=2)


scores_sp = [-35.2, 6.4, 8.5, 8.3, 7.55, 7.75, 6.38, 4.76, 7.0]
times_sp = [0, 4580, 9215, 13822, 18406, 23000, 27600, 32100, 36700]
times_sp = [x / 3600 for x in times_sp]
std_sp = 1.3

ax4.hlines(14.3, 0, 10, label="Human-data")
ax4.fill_between(
    [0, 10],
    np.array([14.3, 14.3]) - std_sp * 1.96,
    np.array([14.3, 14.3]) + std_sp * 1.96,
    alpha=alpha1,
    edgecolor="blue",
)
ax4.plot(times_sp, scores_sp, marker, label="Self-play", color="#ff7f0e")
ax4.fill_between(
    times_sp,
    np.array(scores_sp) - std_sp * 1.96,
    np.array(scores_sp) + std_sp * 1.96,
    alpha=alpha1,
    edgecolor="orange",
)
ax4.hlines(3.0, 0, 10, label="Rollout", color="#2ca02c")
ax4.fill_between(
    [0, 10],
    np.array([3.0, 3.0]) - std_sp * 1.96,
    np.array([3.0, 3.0]) + std_sp * 1.96,
    alpha=alpha1,
    edgecolor="green",
)


# ax4.set_title("Progression of the average score while training (simulations=10).")
ax4.set_ylabel("Score Difference")
ax4.set_xlabel("Training time (hours)")
ax4.legend(loc=5)
fig1.savefig("result_1.png", bbox_inches="tight", dpi=300)
fig2.savefig("result_2.png", bbox_inches="tight", dpi=300)
fig3.savefig("result_3.png", bbox_inches="tight", dpi=300)
fig4.savefig("result_4.png", bbox_inches="tight", dpi=300)

# OLD
# X_10_5 = [200, 130, 100, 50, 25, 10]
# y_10_5 = [4.6, 7.8, 8.6, 8.6, 8.5, 6.8]
# std_10_5 = 0.6

# X_50_5 = [3000, 1000, 500, 300, 200, 130, 50, 25, 10]
# y_50_5 = [15.0, 28.8, 33.0, 32.7, 35.2, 35.1, 30, 28.5, 22.5]
# std_50_5 = 1.8

# X_200_5 = [50, 100, 200, 200, 300, 500, 1000, 2000, 4000, 6000]
# y_200_5 = [41.2, 46.5, 46.2, 46.4, 47.1, 46.0, 45.9, 46.8, 43.8, 37.1]
# std_200_5 = 1.8

# X_500_5 = [12000, 6000, 3000, 1000, 500, 300, 100, 25]
# y_500_5 = [50.1, 51.6, 52.6, 49.5, 51.2, 49.3, 49.3, 36.8]
# std_500_5 = 1.8


# marker = ".-"

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
# ax1.plot(X_10_5, y_10_5, marker, label="10 steps")
# ax1.fill_between(X_10_5, np.array(y_10_5) - std_10_5 * 1.96, np.array(y_10_5) + std_10_5 * 1.96, alpha=0.2)

# ax1.plot(X_50_5, y_50_5, marker, label="50 steps")
# ax1.fill_between(X_50_5, np.array(y_50_5) - std_50_5 * 1.96, np.array(y_50_5) + std_50_5 * 1.96, alpha=0.2)

# ax1.plot(X_200_5, y_200_5, marker, label="200 steps")
# ax1.fill_between(X_200_5, np.array(y_200_5) - std_200_5 * 1.96, np.array(y_200_5) + std_200_5 * 1.96, alpha=0.2)

# ax1.plot(X_500_5, y_500_5, marker, label="500 steps")
# ax1.fill_between(X_500_5, np.array(y_500_5) - std_500_5 * 1.96, np.array(y_500_5) + std_500_5 * 1.96, alpha=0.2)

# ax1.set_title("Effect of different number of steps with different \n C values on the average score (sims=5)")
# # ax1.set_title("Average Score vs C value (Exploration Rate) \n for different number of steps and sims=5")
# # ax1.set_
# ax1.set_xscale("log")
# ax1.set_xlabel("C value (Exploration Rate)")
# ax1.set_ylabel("Average Score")
# ax1.legend()

# sims_step10_c1 = [200, 100, 50, 25, 15, 8, 5, 3]
# time_step10_c1 = [345, 172, 91, 47.6, 30.4, 17.54, 13.8, 8.48]
# score_step10_c1 = [12.8, 11.4, 10.3, 8.4, 7.8, 7.5, 6.1, 4.8]
# std_step10_c1 = 0.6

# steps_sims5_c1 = [10, 50, 200, 500]
# score_sims5_c1 = [8.6, 35.2, 47.1, 51.2]
# time_sims5_c1 = [12.6, 49.9, 193.2, 492.3]

# sims__steps200_c300 = [20, 10, 4, 3, 1]
# score_steps200_c300 = [47.5, 46.7, 44.4, 44.0, 42]
# time_steps200_c300 = [693.0, 353.0, 170.3, 142.0, 71.9]

# marker2 = "-"
# ax2.plot(time_step10_c1, score_step10_c1, marker2, label="Sims with steps=10, c=1")
# ax2.plot(time_sims5_c1, score_sims5_c1, marker2, label="Steps with sims=5, c=optimal")
# ax2.plot(time_steps200_c300, score_steps200_c300, marker2, label="Sims with steps=200, c=300")

# number_base = 130
# number_scale = 10
# for xp, yp, m in zip(time_step10_c1, score_step10_c1, sims_step10_c1):
#     ax2.scatter(
#         xp,
#         yp,
#         marker=f"${m}$",
#         s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
#         color="black",
#         zorder=10,
#     )

# for xp, yp, m in zip(time_sims5_c1, score_sims5_c1, steps_sims5_c1):
#     ax2.scatter(
#         xp,
#         yp,
#         marker=f"${m}$",
#         s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
#         color="black",
#         zorder=10,
#     )

# for xp, yp, m in zip(time_steps200_c300, score_steps200_c300, sims__steps200_c300):
#     ax2.scatter(
#         xp,
#         yp,
#         marker=f"${m}$",
#         s=number_base * (np.floor(np.emath.logn(number_scale, m)) + 1),
#         color="black",
#         zorder=10,
#     )
# ax2.set_xscale("log")
# ax2.set_yscale("log")
# ax2.set_title(
#     "Effect of different number of simulations compared to different \n number of steps on the average score and thinking time"
# )
# # ax2.set_title("Score vs Think time \n for different steps and sims settings")
# ax2.set_xlabel("Think time per move (ms)")
# ax2.set_ylabel("Average Score")
# ax2.legend()


# scores_no_sims = [-30.2, -13.7, -11.5, -8.1, -3.9, 1.3, 4.7, 6.0, 10.8, 16.3, 17.7, 19.8, 19.1, 20.1, 14.5, 16.8, 18.2]
# times_no_sims = [0, 4.2, 8.4, 12.6, 16.8, 21, 25.2, 29.4, 33.6, 37.8, 42, 46.2, 50.4, 54.6, 58.8, 63, 67.2]
# std_no_sims = 1.8

# scores_sims = [-40.5, 22.6, 18.9, 17.9, 14.7, 16.1, 14.9, 17.1, 15.9, 18.1, 18.5, 16.8]
# times_sims = [0, 1.0, 2.9, 4.8, 9.6, 14.4, 19.2, 24.0, 28.8, 33.6, 38.4, 43.2]
# std_sims = 1.8

# scores_boosted = [21.0, 17.5, 17.2, 16.6, 17.4, 15.3]
# times_boosted = [0, 4.1, 8.2, 12.3, 16.4, 20.5]
# std_boosted = 1.8

# ax3.plot(times_no_sims, scores_no_sims, marker, label="RL without sims")
# ax3.fill_between(
#     times_no_sims,
#     np.array(scores_no_sims) - std_no_sims * 1.96,
#     np.array(scores_no_sims) + std_no_sims * 1.96,
#     alpha=0.2,
# )
# ax3.plot(times_sims, scores_sims, marker, label="RL with sims")
# ax3.fill_between(
#     times_sims, np.array(scores_sims) - std_sims * 1.96, np.array(scores_sims) + std_sims * 1.96, alpha=0.2
# )
# ax3.plot(times_boosted, scores_boosted, marker, label="RL boosted")
# ax3.fill_between(
#     times_boosted,
#     np.array(scores_boosted) - std_boosted * 1.96,
#     np.array(scores_boosted) + std_boosted * 1.96,
#     alpha=0.2,
# )
# ax3.set_title("Progression of the average score while training")
# ax3.set_ylabel("Average Score")
# ax3.set_xlabel("Training time (hours)")
# ax3.legend()
# plt.show()
