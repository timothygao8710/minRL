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
      "temperature": 1,
      "beta": 0.9,
    }
)

config = wandb.config

@jax.jit
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
        return nn.Dense(2)(x)

def baseline(states, extra_info):
    return np.average(extra_info['rewards']) # a better baseline is previous pole angle

def compute_adv(rewards, extra_info):
    beta = config.beta # reduces variance at cost of bias
    
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
        S.append(state[0])
        
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
        
@jax.jit
def loss(states, acts, adv, model, params):
    states, acts, adv = jnp.asarray(states), jnp.asarray(acts), jnp.asarray(adv) # convert to jnp arrays
    logits = model.apply(params, states) # pi(s)
    log_probs = nn.log_softmax(logits) # pi(s) -> log p(*|s)
    log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze() # log p(*|s) -> log p(a|s)
    res = log_probs * adv # log p(a|s) -> adv * log p(a|s)
    return jnp.mean(res) # 1/N * sum(adv * log p(a|s))


if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    key = jax.random.PRNGKey(SEED)
    env.reset(seed=SEED)
    pi = PI()
    params = pi.init(key, preprocess(env.render())[None, :])
    lr = config.learning_rate

    dloss_dparams = jax.grad(loss, argnums=4)

    # grad_params E_traj[return] = E_traj[grad_params log P(traj) * return] 
    # --> E_step[grad_params log P(a | s) * return] --> E_step[grad_params log P(a | s) * adv] (rewards to-go + baseline)
    for epoch in trange(1, config.epochs + 1):
        S, A, R = [], []
        
        # 1) trajectory monte carlo sampling
        for sample in range(1, config.rollouts_per_epoch + 1): 
            episode = rollout(key, env)
            key = nxt_key(key)
            episode['rewards'] = compute_adv(episode['rewards'])
            S.extend(episode['states']) # retrieve all images from state
            A.extend(episode['actions'])
            R.extend(episode['rewards'])
            
            if (epoch * config.rollouts_per_epoch + sample) % 100 == 0:
                video = np.stack(episode['states']).astype(np.uint8)
                wandb.log({"rollout_video": wandb.Video(video, fps=30, format="mp4")}, step=epoch)
        
        # 2) update policy
        adv = R - baseline(S, {"rewards":R}) # substract mean reward from episode
        
        grad = dloss_dparams(S, A, adv, pi, params)
        
        params = jax.tree_util.tree_map(lambda u, v: u + lr * v, params, grad) # take direction of gradient step to maximize loss
        
        # 3) logging
        stats = {
            "alive_time": len(S) / config.rollouts_per_epoch,
            "avg_reward": np.mean(R),
            "var_reward": np.variance(R),
            "left_percentage" : A.count(0) / len(A),
            "epoch": epoch,
        }
        wandb.log(stats, step=epoch)

    env.close()
    wandb.finish()
