import wandb
import os, operator
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import jax, jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn
from tqdm import trange
from multiprocessing.pool import Pool
from PIL import Image
import optax

os.environ["SDL_AUDIODRIVER"] = "headless"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

'''
Compared to reinforce_1, we
- Use an optimizer
- 
'''

wandb.init(
    entity="timothy-gao",
    project="RL",
    # mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
      "epochs": 10000,
      "rollouts_per_epoch": 64,
      "max_steps": 500,
      "lr": 1e-2 / 2,
      "discount": 0.99,
      "n_frames": 2,
      "epoch_record_freq": 100,
      "seed": 42,
      "img_size": 64,
    }
)

config = wandb.config

@jax.jit
def preprocess(img, out=config.img_size):
    img = jimage.resize(img, (out, out, 1), method='nearest')
    img = (img - 127.5) / 255.0
    return img

def discount_rewards(rewards):
    for i in range(len(rewards)-2, -1, -1): # reward to-go
        rewards[i] += config.discount * rewards[i+1]
    return rewards

def compute_reward(prev_state, action, next_state):
    return 1

class PI(nn.Module):
    latent_dim: int = 256
    
    @nn.compact
    def __call__(self, S):
        assert(len(S.shape)>=3)
        
        if len(S.shape) == 3: # Add batch dim
            S = S[None, :, :, :] # B, H, W, C
        
        for f in (32,64): 
            S = nn.Conv(f,(4,4),(2,2),padding="SAME")(S)
            S = nn.relu(S)
        
        S = S.reshape((S.shape[0], -1))
        
        for h in (self.latent_dim,96,72):
            S = nn.Dense(h)(S)
            S = nn.relu(S)
        
        return nn.Dense(2)(S)

def rollout(key, env): # returns a single traj = [states, actions, rewards]
    env = env or gym.make("CartPole-v1", render_mode="rgb_array") # create new env if none passed in
    obs, info = env.reset(seed=int(jax.random.randint(key, (), 0, 2**31 - 1)))
    state = preprocess(env.render()), obs
    
    buffer = [state[0]] * config.n_frames # buffer shape (n_frames, H, W, 1)
    S, A, R = [], [], []
    
    for t in range(config.max_steps):
        buffer.append(state[0]) # add frame to buffer
        buffer = buffer[1:] # only keep the most recent n_frames

        S.append(jnp.concat(buffer, axis=2)) # (n_frames, H, W, 1) -> (H, W, n_frames)
        
        logits = pi.apply(params, S[-1]).squeeze() # (1, 2) output -> squeeze -> (2)
        key, subkey = jax.random.split(key, 2) # split key to sample action
        action = int(jax.random.categorical(key=subkey, logits=logits)) # jax.random.categorical directly samples from logits
        
        A.append(action)
        
        obs, _, terminated, truncated, info = env.step(action) # apply action, advance environment
        nxt_state = preprocess(env.render()), obs # get next state
        
        R.append(compute_reward(state, action, nxt_state)) # compute reward
        
        if terminated or truncated:
            break

        state = nxt_state

    return jnp.asarray(S), jnp.asarray(A), jnp.asarray(discount_rewards(R))
        
@jax.jit
def loss(params, states, acts, adv, num_trajs):
    logits = pi.apply(params, states) # pi(s)
    log_probs = nn.log_softmax(logits) # pi(s) -> log p(*|s)
    log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze() # log p(*|s) -> log p(a|s)
    res = log_probs * adv # log p(a|s) -> adv * log p(a|s)
    # return 1 / num_trajs * jnp.sum(res) # 1/|D| * sum(adv * log p(a|s))
    return jnp.mean(res) # 1/|D| * sum(adv * log p(a|s))

@jax.jit
def upd(params, grad, opt, opt_state):
    
    return jnp.mean(res) # 1/|D| * sum(adv * log p(a|s))

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env,
        video_folder="/home/timothygao/minRL/videos",
        name_prefix="epoch",
        episode_trigger=lambda ep: ep % config.epoch_record_freq == 0,
    )
    
    n_rollouts = config.rollouts_per_epoch
    
    key = jax.random.key(config.seed)
    env.reset(seed=int(jax.random.randint(key, (), 0, 2**31 - 1)))
    pi = PI()

    params = pi.init(key, jnp.zeros((1, config.img_size, config.img_size, config.n_frames)))
    dloss_dparams = jax.jit(jax.value_and_grad(loss, argnums=0))
    
    opt = optax.adam(config.lr)
    opt_state = pi_opt.init(params)
    
    for epoch in trange(1, config.epochs + 1):
        S, A, R = [], [], []
        
        # 1) Get monte carlo estimate of raw rollouts
        key, *subkeys = jax.random.split(key, n_rollouts + 1)
        rollout(key, env) # for recording purposes
        for sample in trange(n_rollouts):
            episode = rollout(subkeys[sample], None)
            S.append(episode[0])
            A.append(episode[1])
            R.append(episode[2])
            
        S, A, R = jnp.concat(S, axis=0), jnp.concat(A, axis=0), jnp.concat(R, axis=0)
        adv = R - jnp.mean(R) # Let's use a simple mean baseline
        
        prev_value, grad = dloss_dparams(params, S, A, adv, n_rollouts) # computes monte carlo estimate for grad_params E_traj[return]
        params = jax.tree_util.tree_map(lambda u, v: u + config.lr * v, params, grad) # take direction of gradient step to maximize loss
        nxt_value, grad = dloss_dparams(params, S, A, adv, n_rollouts) # check to make sure nxt_val - prev_val > 0
        
        # 3) logging
        stats = {
            "diff": nxt_value - prev_value,
            "mean_alive_time": len(S) / n_rollouts,
            "mean_reward": jnp.mean(R),
            "var_reward": jnp.var(R),
            "var_return": jnp.sqrt(jnp.var(R) / (n_rollouts - 1)),
            "left_percentage" : jnp.sum(A == 0) / len(A),
            "epoch": epoch,
            "episode": epoch * n_rollouts,
        }
        wandb.log(stats, step=epoch * n_rollouts)

    env.close()
    wandb.finish()
