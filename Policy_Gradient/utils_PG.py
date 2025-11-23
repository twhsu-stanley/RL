import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_frozenlake_REINFORCE(agent, **kwargs):
        """
        Plot value function and corresponding optimal policy on the state space
        """

        grid_size = agent.env.unwrapped.desc.shape
        desc = agent.env.unwrapped.desc.astype(str)

        fig, ax = plt.subplots(figsize=(5,5))
        
        # Grid the state space
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if desc[i,j] == 'S':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'H':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'G':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax.add_patch(rect)
        
        # action arrows: 0: Left, 1: Down, 2: Right, 3: Up
        arrow_dict = {0: (-0.4, 0), 1: (0, 0.4), 2: (0.4, 0), 3: (0, -0.4)} # upside down y-axis
        
        # Write value function in the center of each box
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = i * grid_size[1] + j
                # Value function numerical values
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # Convert to one-hot vector
                state = torch.nn.functional.one_hot(state.long().squeeze(), num_classes=grid_size[0]*grid_size[1]).float()
                
                # Policy arrows
                with torch.no_grad():
                    action_dist = torch.softmax(agent.policy_net.forward(state), dim=-1).detach().numpy()

                for a in range(agent.dim_action):
                    action = action_dist[a]
                    arrow_length = action / 2 # scale down for better visualization

                    if arrow_length < 0.025:
                        continue # skip small probabilities for better visualization

                    dx, dy = arrow_dict[a] 
                    if desc[i,j] in ['H','G']: # hole or goal
                        continue
                    ax.arrow(j + 0.5, i + 0.5, dx * arrow_length, dy * arrow_length, head_width=0.1, head_length=0.1, fc='red', ec='red')

        # Set limits and labels
        ax.set_xlim(0, grid_size[1])
        ax.set_ylim(0, grid_size[0])
        ax.set_aspect('equal')
        ax.invert_yaxis()  # top-left is (0,0)
        ax.set_xticks(np.arange(grid_size[1]+1))
        ax.set_yticks(np.arange(grid_size[0]+1))
        ax.grid(False)
        ax.set_title("Trained Policy: {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

        plt.show()

def plot_frozenlake_ACTOR_CRITIC(agent, **kwargs):
        """
        Plot value function and corresponding optimal policy on the state space
        """

        grid_size = agent.env.unwrapped.desc.shape
        desc = agent.env.unwrapped.desc.astype(str)

        fig1, ax1 = plt.subplots(figsize=(5,5))
        fig2, ax2 = plt.subplots(figsize=(5,5))
        
        # Grid the state space
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if desc[i,j] == 'S':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'H':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'G':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax1.add_patch(rect)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if desc[i,j] == 'S':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'H':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black', alpha=0.35)
                elif desc[i,j] == 'G':
                    rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black', alpha=0.35)
                else:
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black')
                ax2.add_patch(rect)
        
        # action arrows: 0: Left, 1: Down, 2: Right, 3: Up
        arrow_dict = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)} # upside down y-axis
        
        # Write value function in the center of each box
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state = i * grid_size[1] + j
                # Value function numerical values
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # Convert to one-hot vector
                state = torch.nn.functional.one_hot(state.long().squeeze(), num_classes=grid_size[0]*grid_size[1]).float()
                value = round(agent.value_net(state).item(), 2) # Bellman optimality of Q
                ax1.text(j + 0.5, i + 0.5, str(value), color='blue',
                        fontsize=12, ha='center', va='center')
                
                # Policy arrows
                with torch.no_grad():
                    action_dist = torch.softmax(agent.policy_net.forward(state), dim=-1).detach().numpy()

                for a in range(agent.dim_action):
                    action = action_dist[a]
                    arrow_length = action / 2 # scale down for better visualization

                    if arrow_length < 0.025:
                        continue # skip small probabilities for better visualization

                    dx, dy = arrow_dict[a] 
                    if desc[i,j] in ['H','G']: # hole or goal
                        continue
                    ax2.arrow(j + 0.5, i + 0.5, dx * arrow_length, dy * arrow_length, head_width=0.1, head_length=0.1, fc='red', ec='red')

        # Set limits and labels
        ax1.set_xlim(0, grid_size[1])
        ax1.set_ylim(0, grid_size[0])
        ax1.set_aspect('equal')
        ax1.invert_yaxis()  # top-left is (0,0)
        ax1.set_xticks(np.arange(grid_size[1]+1))
        ax1.set_yticks(np.arange(grid_size[0]+1))
        ax1.grid(False)
        ax1.set_title("V(s): {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

        # Set limits and labels
        ax2.set_xlim(0, grid_size[1])
        ax2.set_ylim(0, grid_size[0])
        ax2.set_aspect('equal')
        ax2.invert_yaxis()  # top-left is (0,0)
        ax2.set_xticks(np.arange(grid_size[1]+1))
        ax2.set_yticks(np.arange(grid_size[0]+1))
        ax2.grid(False)
        ax2.set_title("Trained Policy: {}, is_slippery={}".format(kwargs.get("algorithm", ""), kwargs.get("is_slippery", False)))

        plt.show()
    
def plot_evaluation_return(evaluation_return, **kwargs):

    evaluation_return = np.array(evaluation_return)
    evaluation_return_mean = np.mean(evaluation_return, axis=0)
    evaluation_return_std = np.std(evaluation_return, axis=0)

    plt.figure()
    plt.plot(evaluation_return_mean)
    plt.fill_between(range(len(evaluation_return_mean)), evaluation_return_mean - evaluation_return_std, evaluation_return_mean + evaluation_return_std, alpha=0.2)
    plt.grid()
    plt.xlabel("Episodes")
    plt.ylabel("G_0")
    plt.title(f"{kwargs.get("algorithm","")}: Evaluation G_0 over Episodes")
    plt.show()