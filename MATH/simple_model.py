import numpy as np
import os

V_SEC = 10
V_SIZE = 30000 + V_SEC
np.random.seed(0)
logits_proper = np.random.normal(loc=0.0, scale=1, size=10)
#logits_proper = np.array([0 - 0.5 * i for i in range(V_SEC)])

def predict(total_step, init_error_rate, temperature, factor=1.5, proper_logits=logits_proper):
    error_rate = init_error_rate
    entropies_error = []
    entropies_correct = []
    num_correct = 0
    for i in range(total_step):
        if np.random.rand() < error_rate:
            error_rate = 1 - (1 - error_rate) ** factor
        else:
            error_rate = max(error_rate ** factor, init_error_rate)
            num_correct += 1
        entropies_error.append(-error_rate * np.log(error_rate / (V_SIZE - V_SEC)))
        if (1 - error_rate) < 1e-10:
            entropies_correct.append(0)
        else:
#            logit_top = 1 / temperature
#            logit_sec = 0       
#            prob_top = np.exp(logit_top) / (np.exp(logit_top) * V_TOP + np.exp(logit_sec) * V_SEC)
#            prob_sec = np.exp(logit_sec) / (np.exp(logit_top) * V_TOP + np.exp(logit_sec) * V_SEC)
#            entropies_correct.append(-(1 - error_rate) * (prob_top * np.log(((1 - error_rate) * prob_top)) * V_TOP + prob_sec * np.log((1 - error_rate) * prob_sec) * V_SEC))
            prob = (1 - error_rate) * np.exp(proper_logits / temperature) / np.sum(np.exp(proper_logits / temperature))
            entropies_correct.append(prob @ (-np.log(prob)))

    return np.mean(entropies_error), np.mean(entropies_correct), num_correct / total_step

def cal_init_error_rate(temp):
    #logit_top = 1 / temp
    #logit_sec = 0
    logit_nonsense = -10 / temp
    #prob_nonsense = np.exp(logit_nonsense) / (np.exp(logit_top) * V_TOP + np.exp(logit_sec) * V_SEC + np.exp(logit_nonsense) * V_SIZE)
    logits_all = np.concatenate([logits_proper / temp, [logit_nonsense] * (V_SIZE - V_SEC)])
    #softmax
    prob_all = np.exp(logits_all) / np.sum(np.exp(logits_all)) 
    prob_nonsense = prob_all[-1]
    init_error_rate = prob_nonsense * (V_SIZE - V_SEC)
    return init_error_rate

def main(factor=2, save_path='plots/toy_model'):
    temperatures = [0.1 + 0.01 * i for i in range(120)]
    init_error_rates = [cal_init_error_rate(temp) for temp in temperatures]
#    log_init_error_rates =  [-4.5 + 0.1 * i for i in range(30)]
#    init_error_rates = np.exp(log_init_error_rates)
    trial = 500
    total_entropy_list = []
    total_correct_rate_list = []
    for i in range(len(init_error_rates)):
        init_error_rate = init_error_rates[i]
        temperature = temperatures[i]
        total_entropy = 0
        total_entropy_error = 0
        total_entropy_correct = 0
        total_correct_rate = 0
        for i in range(trial):
            entropies_error, entropies_correct, correct_rate = predict(512, init_error_rate, temperature=temperature, factor=factor)
            total_entropy_error += entropies_error
            total_entropy_correct += entropies_correct
            total_entropy += entropies_error + entropies_correct
            total_correct_rate += correct_rate
        total_entropy_list.append(total_entropy / trial)
        total_correct_rate_list.append(total_correct_rate / trial)
        print(f'Error Rate: {init_error_rate}, Entropy: {total_entropy / trial}, Error Entropy: {total_entropy_error / trial}, Correct Entropy: {total_entropy_correct / trial}')
    log_entropy = np.log(total_entropy_list)
    # plot curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(init_error_rates, total_entropy_list, linestyle='-', marker='o', label=f'alpha={factor}, N0=5, N1=1000, l0=0, l1=-5, sigma=1', markersize=5)
    plt.xlim(0, 0.3)
    plt.xlabel('Initial Error Rate', fontsize=16)
    plt.ylabel('Entropy', fontsize=16)
    plt.title(f'Initial Error Rate v.s. Entropy in Simulation', fontsize=16)
    #plt.legend(fontsize=13)
    plt.grid('y')
    #plt.savefig(f'simple_model_{factor}.png')
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_error_rate_vs_entropy.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(init_error_rates, 1 - np.array(total_correct_rate_list), linestyle='-', marker='o', label=f'alpha={factor}, N0=5, N1=1000, l0=0, l1=-5, sigma=1', markersize=5)
    #plt.xscale('log')
    plt.xlim(0, 0.5)
    plt.xlabel('Initial Error Rate', fontsize=16)
    plt.ylabel('Improper Tokens (%)', fontsize=16)
    plt.title(f'Initial Error Rate v.s. Improper Tokens in Simulation', fontsize=16)
    #plt.legend(fontsize=13)
    plt.grid('y')
    #plt.savefig(f'simple_model_{factor}_Tokens.png')
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_error_rate_vs_improper_token.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(temperatures, log_entropy, linestyle='-', marker='o', label=f'alpha={factor}, N0=5, N1=30000, l0=0, l1=-10, sigma=1', markersize=5)
    plt.xlabel('Temperature', fontsize=16)
    plt.ylabel('Log Entropy', fontsize=16)
    plt.grid('y')
    #plt.legend(fontsize=13)
    plt.title(f'Temperature v.s. Log Entropy in Simulation', fontsize=16)
    #plt.savefig(f'simple_model_{factor}_log_log.png')
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_temperature_vs_log_entropy.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(temperatures, total_entropy_list, linestyle='-', marker='o', label=f'alpha={factor}, N0=5, N1=30000, l0=0, l1=-10, sigma=1', markersize=5)
    plt.xlabel('Temperature', fontsize=16)
    plt.ylabel('Entropy', fontsize=16)
    plt.grid('y')
    #plt.legend(fontsize=13)
    plt.title(f'Temperature v.s. Entropy in Simulation', fontsize=16)
    #plt.savefig(f'simple_model_{factor}_log_init_rate.png')
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_temperature_vs_entropy.png'))

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(10,5))

    # calculate log_entropy's derivative respect to temperature
    # due to variance, we select the first time where the derivative is positive and the last time where the derivative is negative, and take the average
    log_entropy_derivative = np.gradient(log_entropy, temperatures)
    derivative2 = np.gradient(log_entropy_derivative, temperatures)
    print(derivative2)
    first_turn_point = temperatures[np.argmax(derivative2 > 0)]
    print(first_turn_point)
    # Plot log_entropy on the first (left) y-axis
    color_left = 'tab:blue'
    ax1.set_xlabel('Temperature', fontsize=16)
    ax1.set_ylabel('Entropy', color=color_left, fontsize=16)
    ax1.plot(temperatures, total_entropy_list, 
            color=color_left, marker='o', linestyle='-',
            label=f'Entropy (alpha={factor})', markersize=5)
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid(True, axis='y')

    # Create a twin Axes sharing the x-axis to plot total_entropy_list
    ax2 = ax1.twinx()
    color_right = 'tab:red'
    ax2.set_ylabel('Log Entropy', color=color_right, fontsize=16)
    ax2.plot(temperatures, log_entropy, 
            color=color_right, marker='o', linestyle='-',
            label=f'Log Entropy (alpha={factor})', markersize=5)
    ax2.tick_params(axis='y', labelcolor=color_right)

    # Create a combined legend
    #We can extract the line objects from both axes and combine them
    turning = ax1.axvline(x=first_turn_point, color='green', linestyle='--', label=f'Turning Point: {first_turn_point:.2f}')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=13, loc='best')
    plt.title('Temperature vs. Entropy and Log Entropy in Simulation', fontsize=16)

    # Save the figure
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_merged.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(temperatures, 1 - np.array(total_correct_rate_list), linestyle='-', label="Temperature v.s. Improper Tokens", marker='o', markersize=5)
    plt.xlabel('Temperature', fontsize=16)
    plt.ylabel('Improper Tokens (%)', fontsize=16)
    plt.grid('y')
    plt.axvline(x=first_turn_point, color='green', linestyle='--', label=f'Turning Point: {first_turn_point:.2f}')
    plt.legend(fontsize=13)
    plt.ylim(0, 0.2)
    plt.title(f'Temperature v.s. Improper Tokens in Simulation', fontsize=16)
    #plt.savefig(f'simple_model_{factor}_T-I.png')
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_temperature_vs_improper_token.png'))

    # temperature vs. init_error_rate
    plt.figure(figsize=(10, 5))
    plt.plot(temperatures, init_error_rates, linestyle='-', label=f'alpha={factor}, N0=5, N1=30000, l0=0, l1=-10, sigma=1', markersize=5)
    plt.xlabel('Temperature', fontsize=16)
    plt.ylabel('Initial Error Rate', fontsize=16)
    plt.ylim(0, 0.2)
    plt.grid('y')
    #plt.legend(fontsize=13)
    plt.title(f'Temperature v.s. Initial Error Rate in Simulation', fontsize=16)
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_temperature_vs_init_error_rate.png'))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(top=0.85, bottom=0.15)

    # Plot on the primary (bottom) x-axis
    line1 = ax1.plot(
        init_error_rates, 
        1 - np.array(total_correct_rate_list), 
        marker='o',
        linestyle='-',
        color='tab:blue',
        label='Initial Error Rate'
    )
    ax1.set_xlabel('Initial Error Rate', fontsize=16, color='tab:blue')
    # color the x-axis
    ax1.tick_params(axis='x', colors='tab:blue')
    ax1.set_ylabel('Improper Tokens (%)', fontsize=16)

    # Create a second x-axis on the top
    ax2 = ax1.twiny()

    line2 = ax2.plot(
        temperatures, 
        1 - np.array(total_correct_rate_list), 
        marker='o',
        linestyle='-',
        color='tab:red',
        label='Temperature'
    )
    ax2.axvline(x=first_turn_point, color='green', linestyle='--', label=f'Turning Point: {first_turn_point:.2f}')
    ax2.set_xlabel('Temperature', fontsize=16, color='tab:red')
    ax2.tick_params(axis='x', colors='tab:red')

    # Combine legends from both axes (if needed)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=13, loc='best')

    ax1.grid(axis='y')
    plt.ylim(0, 0.2)
    plt.title('Initial Error Rate & Temperature vs. Improper Tokens', fontsize=16)
    plt.savefig(os.path.join(save_path, f'simple_model_{factor}_two_x_axes.png'))


    #with open(f'simple_model_{factor}.txt', 'w') as f:
    with open(os.path.join(save_path, f'simple_model_{factor}.txt'), 'w') as f:
        for i in range(len(init_error_rates)):
            f.write(str(init_error_rates[i]) + ' ' + str(total_entropy_list[i]) + '\n')
    
    return temperatures, total_entropy_list, total_correct_rate_list

if __name__ == '__main__':
    save_path = "plots/toy_model"
    os.makedirs(save_path, exist_ok=True)
    x, y0, z0 = main(factor=1.2, save_path=save_path)
    x, y1, z1 = main(factor=1.5, save_path=save_path)
    x, y2, z2 = main(factor=2, save_path=save_path)
    x, y3, z3 = main(factor=2.5, save_path=save_path)
    x, y4, z4 = main(factor=3, save_path=save_path)

    import matplotlib.pyplot as plt
#    plt.plot(x, y0, label='beta=1.2')
    plt.plot(x, y1, label='beta=1.5')
    plt.plot(x, y2, label='beta=2')
    plt.plot(x, y3, label='beta=2.5')
    plt.plot(x, y4, label='beta=3')
    plt.xlabel('Temperature')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'simple_model_all.png'))

    plt.figure()
    plt.plot(x, np.log(y0), label='beta=1.2')
    plt.plot(x, np.log(y1), label='beta=1.5')
    plt.plot(x, np.log(y2), label='beta=2')
    plt.plot(x, np.log(y3), label='beta=2.5')
    plt.plot(x, np.log(y4), label='beta=3')
    plt.xlabel('Temperature')
    plt.ylabel('Log Entropy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'simple_model_all_log.png'))

    plt.figure()
    plt.plot(x, z0, label='beta=1.2')
    plt.plot(x, z1, label='beta=1.5')
    plt.plot(x, z2, label='beta=2')
    plt.plot(x, z3, label='beta=2.5')
    plt.plot(x, z4, label='beta=3')
    plt.xlabel('Temperature')
    plt.ylabel('Correct Rate')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'simple_model_all_correct_rate.png'))