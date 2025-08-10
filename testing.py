import jax
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# from reinforce import rollout, SEED  # adjust import to your module path

def test_rollout_run():
    key = jax.random.PRNGKey(1)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env,
        video_folder="/home/timothygao/minRL/videos",
        name_prefix="rollout",
        episode_trigger = lambda ep: ep % 1 == 0,
    )
  
    obs, info = env.reset()
    state = env.render(), obs
    
    cnt = 0
    for t in range(10000):
        if state[1][2] < 0:
            action = 0
        else:
            action = 1

        obs, _, terminated, truncated, info = env.step(action) # 35
        nxt_state = env.render(), obs
        
        if terminated or truncated:
            break

        state = nxt_state
        cnt += 1
    env.close()
    print(cnt)

if __name__ == "__main__":
    test_rollout_run()
