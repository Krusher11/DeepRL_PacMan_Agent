import matplotlib.pyplot as plt
import pickle
import numpy as np





def load(filepath):
    with open(filepath, 'rb') as file:
        fileInfo = pickle.load(file)

    return fileInfo



def plot_1_graph(file_path):
    file_1 = load(file_path)
    num_episodes = len(file_1)
    episodes = list(range(1,num_episodes + 1))
    plt.figure(figsize=(12, 6))
    plt.scatter(episodes, file_1, label='DDQN', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Testing in-game scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('best_testing_results.jpg')
    score = sum(file_1)
    score = score/num_episodes
    filtered_scores = [score for score in file_1 if score > 500]
    if filtered_scores:
        average_score = sum(filtered_scores) / len(filtered_scores)
        print("Average score of scores above 200:", average_score)
    else:
        print("No scores above 200 found in the file.")
        print(score)

def moving_mean(file_path_1,file_path_2,x_label,y_label,save_name):
    file_1 = load(file_path_1)
    file_2 = load(file_path_2)
    num_episodes = len(file_1[0])
    print(num_episodes)
    num_training_runs = len(file_1)
    print(num_training_runs)

    plotting_info = []

    episode = [run[0] for run in file_1]
    DQN_moving_average_reward = [run[1] for run in file_1]
    DDQN_moving_average_reward = [run[1] for run in file_2]
    plotting_info.append((episode,DQN_moving_average_reward,DDQN_moving_average_reward))


    # Assuming `episodes` and `scores` are your data
    window_size = 10  # Adjust this based on how smooth you want the plot

    # Calculate moving mean and standard deviation
    moving_mean = np.convolve(DQN_moving_average_reward, np.ones(window_size)/window_size, mode='valid')
    moving_std = np.array([np.std(DQN_moving_average_reward[i:i+window_size]) for i in range(len(DQN_moving_average_reward) - window_size + 1)])

    # Calculate x-axis values for the plot
    x_values = range(window_size//2, len(DQN_moving_average_reward) - window_size//2)

    # Plotting
    plt.figure(figsize=(12, 6))

   # plt.plot(x_values, DQN_moving_average_reward, label='Original Scores', alpha=0.5, color='blue')
    plt.plot(x_values, moving_mean, label='Moving Mean', color='green')
    plt.plot(x_values, moving_std, label='Moving Standard Deviation', color='orange')

    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.title('Training Scores with Moving Mean and Standard Deviation')
    plt.grid(True)
    plt.show()
    

def plot_2_graphs(file_path_1,file_path_2,x_label,y_label,save_name):

    file_1 = load(file_path_1)
    file_2 = load(file_path_2)
    num_episodes = len(file_1[0])
    print(num_episodes)
    num_training_runs = len(file_1)
    print(num_training_runs)

    plotting_info = []

    episode = [run[0] for run in file_1]
    DQN_moving_average_reward = [run[4] for run in file_1]
    DDQN_moving_average_reward = [run[4] for run in file_2]
    plotting_info.append((episode,DQN_moving_average_reward,DDQN_moving_average_reward))
  
    print(episode)
    
    




    # Plot episode reward
    # Plot DQN rewards in blue
    # episode,reward_1,reward_2= zip(*plotting_info)
    plt.figure(figsize=(10, 5))
    plt.plot(episode, DQN_moving_average_reward, label='unfin', color='blue')
    plt.plot(episode, DDQN_moving_average_reward, label='finished', color='green')



    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_name)

def calc_ave(file_path_1,file_path_2,file_path_3):
    file_1 = load(file_path_1)
    file_2 = load(file_path_2)
    file_3  = load(file_path_3)

    ave_1 = sum(file_1)/len(file_1)
    ave_2 = sum(file_2)/len(file_2)
    ave_3 = sum(file_3)/len(file_3)
    print(f'Average 1: {ave_1}, Average 2: {ave_2}, Average 3: {ave_3}')
    
def plot_3_graphs(file_path_1,file_path_2,file_path_3,x_label,y_label,save_name):

    file_1 = load(file_path_1)
    file_2 = load(file_path_2)
    file_3  = load(file_path_3)
    num_episodes = len(file_1[0])
    print(num_episodes)
    num_training_runs = len(file_1)
    print(num_training_runs)

    plotting_info = []

    episode = [run[0] for run in file_1]
    results_1 = [run[4] for run in file_1]
    results_2 = [run[4] for run in file_2]
    results_3 = [run[4] for run in file_3]
    #plotting_info.append((episode,DQN_moving_average_reward,DDQN_moving_average_reward))
  
    #print(episode)
    
    




    # Plot episode reward
    # Plot DQN rewards in blue
    # episode,reward_1,reward_2= zip(*plotting_info)
    plt.figure(figsize=(10, 5))
    plt.plot(episode, results_1, label='Linear', color='blue')
    plt.plot(episode, results_2, label='Curve', color='green')
    plt.plot(episode, results_3, label='Constant', color='red')



    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize = 14)
    plt.savefig(save_name)

def print_graph_from_path(path,x_label,y_label, title,save_name):
    
    file_1 = load(path)

    #print(file_1)

    
    num_episodes = len(file_1[0])
    print(num_episodes)
    num_training_runs = len(file_1)
    print(num_training_runs)

    plotting_info = []

    for episode in range(num_training_runs):
        moving_average_reward = [run[episode][4] for run in file_1]
        plotting_info.append((episode,moving_average_reward[0],moving_average_reward[1],moving_average_reward[2]))
  

    




    # Plot episode reward
    # Plot DQN rewards in blue
    episode,reward_1,reward_2,reward_3= zip(*plotting_info)
    plt.figure(figsize=(10, 5))
    plt.plot(episode, reward_1, label='Constant', color='blue')
    plt.plot(episode, reward_2, label='Linear', color='green')
    plt.plot(episode, reward_3, label='Exponential', color='red')


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(save_name)



def plot(x_values,y_values,x_label,y_label, colors):

    pass



def main():

    path_1 = '/home/shankar/Documents/Github/DeepQlearning/Testing/DDQN_agent_ghosttimer_0'
    path_2 = '/home/shankar/Documents/Github/DeepQlearning/Testing/DQN_ghost_0_timer'
    path_3 = '/home/shankar/Documents/Github/DeepQlearning/Testing/DQN_testingQ_100'
    calc_ave(path_1,path_2,path_3)
   
 
if __name__ == '__main__':
    main()