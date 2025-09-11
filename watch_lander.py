# watch_lander.py
import time
import gymnasium as gym
from stable_baselines3 import DQN

def watch(num_episodes=5):
    # Load trained model
    model = DQN.load("dqn_lunarlander_final")

    # Create a fresh environment with a GUI
    env = gym.make("LunarLander-v3", render_mode="human")

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done, total_reward = False, 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(1/60)   # smooth animation (60 FPS)

        print(f"Episode {ep+1} reward: {total_reward:.1f}")

    input("Finished. Press Enter to close...")
    env.close()

if __name__ == "__main__":
    watch()
