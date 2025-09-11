# LL2D.py
import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    os.makedirs("./logs/best", exist_ok=True)
    os.makedirs("./logs/eval", exist_ok=True)

    # Training env (wrapped with Monitor for stats) ---
    train_env = Monitor(gym.make("LunarLander-v3"))

    # Eval env + callbacks: stop at mean >= 200, save best model ---
    eval_env = Monitor(gym.make("LunarLander-v3"))
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=200.0, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best",
        log_path="./logs/eval",
        eval_freq=10_000,         # evaluate every 10k steps
        n_eval_episodes=10,       # average over 10 episodes
        deterministic=True,
        callback_on_new_best=stop_cb,
    )

    model = DQN(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=3e-4, buffer_size=200_000, learning_starts=10_000,
        batch_size=256, gamma=0.99, train_freq=4, gradient_steps=1,
        target_update_interval=1_000, exploration_fraction=0.2, exploration_final_eps=0.02,
        seed=42
    )

    # Will stop early when eval mean >= 200; otherwise runs up to max timesteps
    model.learn(total_timesteps=500_000, callback=eval_cb)
    model.save("dqn_lunarlander_final")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
