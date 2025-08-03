import wandb
import os, operator
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import jax, jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn
from tqdm import trange

SEED = 42

def split_key(root, n):
    return list(jax.random.split(root, n))

def nxt_key(key):
    return split_key(key, 2)[1]

wandb.init(
    entity="timothy-gao",
    project="RL",
    # mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
      "epochs": 1000,
      "rollouts_per_epoch": 50,
      "max_steps": 500,
      "learning_rate": 1e-6,
      "temperature": 1.8,
    }
)

config = wandb.config

def preprocess(img_np: np.ndarray, out: int = 64) -> jnp.ndarray:
    gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
    gray = jnp.asarray(gray, dtype=jnp.float32) / 255.0
    gray = jimage.resize(gray, (out, out), method="linear")
    return gray

class PI(nn.Module):
    latent_dim: int = 256
    @nn.compact
    def __call__(self, S):
        assert(len(S.shape)==3)
        x = S[:, :, :, None]
        for f in (64,128,256):
            x = nn.Conv(f,(4,4),(2,2),padding="SAME")(x)
            x = nn.gelu(x)
        x = x.reshape((x.shape[0], -1))
        for h in (self.latent_dim,192,96,72):
            x = nn.Dense(h)(x); x = nn.gelu(x)
        return nn.log_softmax(nn.Dense(2)(x))

env = gym.make("CartPole-v1", render_mode="rgb_array")
key = jax.random.PRNGKey(SEED)

env.reset(seed=SEED)

pi = PI()
params = pi.init(key, preprocess(env.render())[None, :])

def baseline(state, extra_info):
    pass

def compute_adv(rewards, extra_info):
    beta = extra_info["beta"] # reduces variance at cost of bias
    
    for i in range(len(rewards)-2, -1, -1): # reward to-go
        rewards[i] += beta * rewards[i+1]
    
    return rewards

def get_pole_angle(obs):
    return obs[2]

def compute_reward(s, a, s_nxt):
    return -get_pole_angle(s_nxt[1])

# returns traj = {states, actions, rewards}
def rollout(key, env):
    obs, info = env.reset()
    state = preprocess(env.render()), obs
    
    S, A, R = [], [], []
    
    for t in range(config.max_steps):
        S.append(state)
        
        # get action
        logits = pi.apply(params, state[0][None, :]).squeeze()
        action = int(jax.random.categorical(key=key, logits=logits))
        key = nxt_key(key)
        
        A.append(action)
        
        # change environment, get next state
        obs, _, terminated, truncated, info = env.step(action)
        nxt_state = preprocess(env.render()), obs
        
        # compute reward
        R.append(compute_reward(state, action, nxt_state))
        
        if terminated or truncated:
            break

        state = nxt_state
    
    return {"states": S, "actions": A, "rewards" : R}
        




# for epoch in trange(1, config.epochs + 1):
#     for rollout in range(1, config.rollouts_per_epoch + 1):
        
        
        
        
            
#             logits = logp_apply(params, img_obs)[0]
#             action = int(jax.random.categorical(subkey, logits))
            
#             total_reward += reward
            
#             tot_samples += 1

#             # compute grad & accumulate
#             grad = grad_fn(params, img_obs, action)
#             gn = float(pytree_norm(grad))
#             grad_norms.append(gn)
#             grad_sum = jax.tree_util.tree_map(operator.add, grad_sum, grad)
#             left_fracs.append(float(action == 0))
#             step_cnt += 1
#             if done or truncated:
#                 alive_times.append(step_cnt)
#                 break
        
#         epoch_grad_sum = jax.tree_util.tree_map(lambda a, b: a * total_reward + b, grad_sum, epoch_grad_sum)

#     params = jax.tree_util.tree_map(lambda p,g: p - g / tot_samples,
#                                     params, epoch_grad_sum)

#     stats = {
#       "alive_time": np.mean(alive_times),
#       "alive_variance": np.var(alive_times),
#       "frac_action_left": np.mean(left_fracs),
#       "grad_norm": np.mean(grad_norms),
#       "epoch": epoch,
#     }
    
#     wandb.log(stats, step=epoch)

#     if epoch % 100 == 0:
#         frames = []
#         obs = env.reset()[0]
#         for _ in range(200):
#             frames.append(env.render())
#             img_obs = preprocess(frames[-1])
#             rng_key, subkey = jax.random.split(rng_key)
#             action = int(jax.random.categorical(subkey, logp_apply(params, img_obs)[0]))
#             obs, *_ = env.step(action)
#         video = np.stack(frames).astype(np.uint8)
#         wandb.log({"rollout_video": wandb.Video(video, fps=30, format="mp4")}, step=epoch)

# env.close()
# wandb.finish()
