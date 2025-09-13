# LunarLander DQN Agent

This project trains and evaluates a Deep Q-Network (DQN) agent on the **LunarLander-v3** environment from Gymnasium. The trained agent learns to control a lander to achieve soft landings between the flags.

## Requirements
Install dependencies with:
```bash
pip install "gymnasium[box2d]" stable-baselines3
```

## Environment Details
- **State space:** 8 continuous variables
  -  x position
  -  y position
  -  x velocity
  -  y velocity
  -  lander angle
  -  angular velocity
  -  left leg contact (0 or 1)
  -  right leg contact (0 or 1)  

- **Action space:** 4 discrete actions  
  - 0: Do nothing  
  - 1: Fire left orientation engine  
  - 2: Fire main engine  
  - 3: Fire right orientation engine

## Files
- **LL2D.py** – trains a DQN agent on `LunarLander-v3` with evaluation and model saving.  
- **watch_lander.py** – loads a trained model and plays multiple episodes with live rendering.   
- **dqn_lunarlander_final.zip** – saved trained model weights.


## Usage

### Train the agent
```bash
python LL2D.py
```

This will train for up to 500k timesteps and save the model as `dqn_lunarlander_final.zip`.  
The mean evaluation reward over 20 episodes will be printed after training.

### Watch the trained agent
```bash
python watch_lander.py
```

This opens a window and plays several episodes with the trained agent.  
Rewards for each episode are printed in the terminal.


## Notes
- Training always runs the full number of timesteps unless you add a [StopTrainingOnRewardThreshold callback](https://stable-baselines3.readthedocs.io/en/stable/guide/callbacks.html) to stop early when average reward ≥200.  
- You can keep training for more timesteps to further improve stability.  
- Adjust `num_episodes` in `watch_lander.py` to watch more or fewer runs.
- Logs and best model checkpoints are written to the ./logs/ folder. You can add /logs/ to your .gitignore to keep them out of version control.
