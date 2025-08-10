import wandb
import os, operator
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import jax, jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn
from tqdm import trange

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

SEED = 42

wandb.init(
    entity="timothy-gao",
    project="RL",
    # mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
      "epochs": 100000,
      "rollouts_per_epoch": 200,
      "max_steps": 500,
      "learning_rate": 1e-2,
      "temperature": 0.5,
      "beta": 0.99,
      "history_frames": 2,
      "record_freq": 5000,
    }
)

config = wandb.config

@jax.jit
def preprocess(img_np, out: int = 64):
    gray = jnp.dot(img_np[..., :3], jnp.array([0.299, 0.587, 0.114]))
    gray = jnp.asarray(gray, dtype=jnp.float32) / 255.0
    gray = jimage.resize(gray, (out, out), method="linear", antialias=True)
    return gray

class PI(nn.Module):
    latent_dim: int = 256
    @nn.compact
    def __call__(self, S):
        assert(len(S.shape)>=3)
            
        if len(S.shape) == 3:
            S = S[None, :, :, :] # add channel dim
        
        # print(S.shape) # B, H, W, C
        
        for f in (32,64):
            S = nn.Conv(f,(4,4),(2,2),padding="SAME")(S)
            S = nn.relu(S)
        
        S = S.reshape((S.shape[0], -1))
        
        for h in (self.latent_dim,96,72):
            S = nn.Dense(h)(S)
            S = nn.relu(S)
        
        return nn.Dense(2)(S)

def baseline(states, extra_info=None):
    return jnp.mean(extra_info['rewards']) # a better baseline is previous pole angle

def compute_adv(rewards, extra_info=None):
    beta = config.beta # reduces variance at cost of bias (TODO: GAE)
    
    for i in range(len(rewards)-2, -1, -1): # reward to-go
        rewards[i] += beta * rewards[i+1]
    
    return rewards

def rad_to_deg(x):
    import math
    return x / (2 * math.pi) * 360

def get_pole_angle(obs):
    return rad_to_deg(obs[2])

def compute_reward(s, a, s_nxt):
    # print(get_pole_angle(s[1]))
    # print("Reward", abs(get_pole_angle(s[1])) - abs(get_pole_angle(s_nxt[1])))
    
    # return -abs(get_pole_angle(s_nxt[1]))
    # return abs(get_pole_angle(s[1])) - abs(get_pole_angle(s_nxt[1]))
    return 1

# returns traj = {states, actions, rewards}
def rollout(key, env):
    if env == None:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    obs, info = env.reset()
    state = preprocess(env.render()), obs
    
    buffer = [state[0] for _ in range(config.history_frames)]
    S, A, R = [], [], []
    
    for t in range(config.max_steps):
        buffer.append(state[0])
        buffer = buffer[1:]
        
        # print("Before", jnp.asarray(buffer).shape)
        S.append(jnp.stack(jnp.asarray(buffer), axis=2))
        
        # print("Input", S[-1].shape)
        
        # get action
        logits = pi.apply(params, S[-1]).squeeze()
        key, subkey = jax.random.split(key, 2)
        action = int(jax.random.categorical(key=key, logits=logits))
        
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
def loss(params, states, acts, adv, num_trajs):
    logits = pi.apply(params, states) # pi(s)
    log_probs = nn.log_softmax(logits) # pi(s) -> log p(*|s)
    log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze() # log p(*|s) -> log p(a|s)
    res = log_probs * adv # log p(a|s) -> adv * log p(a|s)
    return jnp.mean(res) # 1/|D| * sum(adv * log p(a|s))

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env,
        video_folder="/home/timothygao/minRL/videos",
        name_prefix="rollout",
        episode_trigger=lambda ep: ep % config.record_freq == 0,
    )
    key = jax.random.PRNGKey(SEED)
    env.reset(seed=SEED)
    pi = PI()
    
    dummy = jnp.zeros((1, 64, 64, config.history_frames), dtype=jnp.float32)

    params = pi.init(key, dummy)
    lr = config.learning_rate

    dloss_dparams = jax.value_and_grad(loss, argnums=0)

    # grad_params E_traj[return] = E_traj[grad_params log P(traj) * return] 
    # --> E_traj[sum over steps (grad_params log P(a | s) * return)] --> E_traj[sum over steps (grad_params log P(a | s) * return)] (rewards to-go + baseline)
    for epoch in trange(1, config.epochs + 1):
        S, A, R = [], [], []
        returns = []
        
        key, *subkeys = jax.random.split(key, config.rollouts_per_epoch + 1)

        for sample in trange(config.rollouts_per_epoch):
            episode = rollout(subkeys[sample], env)
            episode['rewards'] = compute_adv(episode['rewards'])
            S.extend(episode['states'])
            A.extend(episode['actions'])
            R.extend(episode['rewards'])
            returns.append(len(episode['states']))
        
        S, A, R = jnp.asarray(S), jnp.asarray(A), jnp.asarray(R) # convert to jnp arrays
        
        # 2) update policy
        adv = R - baseline(S, {"rewards":R}) # substract mean reward from episode
        # adv = R # we already substracted baseline (current pole angle)
        
        prev_value, grad = dloss_dparams(params, S, A, adv, config.rollouts_per_epoch)
        
        params = jax.tree_util.tree_map(lambda u, v: u + lr * v, params, grad) # take direction of gradient step to maximize loss
        
        nxt_value, grad = dloss_dparams(params, S, A, adv, config.rollouts_per_epoch)
        
        # 3) logging
        stats = {
            "diff": nxt_value - prev_value,
            "mean_alive_time": len(S) / config.rollouts_per_epoch,
            "mean_reward": jnp.mean(R), # estimated mean of distribution over rewards, not grad_params log P(a | s) * adv
            "var_reward": jnp.var(R) * config.rollouts_per_epoch / (config.rollouts_per_epoch - 1), # estimated variance of ^
            "std_error_reward": jnp.sqrt(jnp.var(R) / (config.rollouts_per_epoch - 1)),  # estimated std of sample mean (standard error)
            # Var((X1 + X2 + ... + Xn) / n) = Var(X1) / n. Estimated Var(X1) = E[(R_mean - R)^2] * n/(n-1). Var of esimator -> Var(x1) / (n-1). 
            "left_percentage" : jnp.sum(A == 0) / len(A),
            "epoch": epoch,
        }
        wandb.log(stats, step=epoch)

    env.close()
    wandb.finish()
