
from jax.flatten_util import ravel_pytree
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

os.environ["SDL_AUDIODRIVER"] = "headless"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

'''
Solving the CartPole-v1 with only image observations

The main challenge is low signal to noise ratio

Builds on 1_reinforce with:
- Use value function for baseline (optimal control variate)
- GAE (higher bias for lower variance) 

For unbiased V, the TD residual a * V(s_i+1) + r_i - V(s_i) is unbiased estimator for a-just advantage

Notice sum 0 to T of a^i (a * V(s_i+1) + r_i - V(s_i)) = -V(s_0) + r_0 + a * r_1 + a^2 * r_2 + ... + a^T * r_T + a^T+1 V(s_T+1) 

In practice V is biased,

larger T <-> 

Higher variance (more r_i terms, sum of r.v.'s -> summed variances) & 
Lower bias (rewards are unbiased, V which contributes bias is more heavily discounted)

Notice that using discounted rewards does the same thing: trades off variance for bias. One clean way to do this is consider the 
exponentially weight average. The tradeoff is controlled by a decay rate a, instead of hardcoding horizon length.

We do the same thing here, let A(T) = sum 0 to T (a)^i (TD resid_i)
Then, we take exponentially weight average of A(T)s, with decay rate b. Notice this is still a-just for unbiased V.

b = 0: -V(s_i) + r_0 - a * V(s_i+1)

0 < b < 1: roughly controls horizon length (small constant * 1/(1-b)) / how "far out" / discounted our V is

b = 1: -V(s_i) + r_0 + a * r_1 + a^2 * r_2 + ...

turns out, after multiplying by common factor (1 - b), discounted A(T)s simplifies to a nice clean formula 

sum (ab)^i (Td residual_i)
'''

wandb.init(
    entity="timothy-gao",
    project="RL",
    # mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
      "epochs": 10000,
      "rollouts_per_epoch": 50,
      "max_steps": 500,
      "lr": 1e-3,
      "a": 0.99, # discount on rewards
      "b": 0.98, # discount on TD residual, used in GAE
      "n_frames": 2,
      "epoch_record_freq": 100,
      "seed": 42,
      "img_size": 64,
    }
)

config = wandb.config

class NN(nn.Module):
    out_dim: int
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
        
        return nn.Dense(self.out_dim)(S)

@jax.jit
def preprocess(img, out=config.img_size):
    img = jimage.resize(img, (out, out, 1), method='nearest')
    img = (img - 127.5) / 255.0
    return img

@jax.jit
def comp_adv(R, V, a, b): # GAE https://arxiv.org/abs/1506.02438
    # comp td-resids
    resids = R - V + a * jnp.concat([V[1:], jnp.zeros(1)])
    return suf_sum_decay(resids, a * b)

@jax.jit # get suffix sum with decay
def suf_sum_decay(vals, decay_coef): 
    
    def roll(carry_i, input_i): # (carry_i, input_i) -> (carry_i+1, output_i)
        return input_i + decay_coef * carry_i, input_i + decay_coef * carry_i
    
    _, vals = jax.lax.scan(roll, 0.0, vals[::-1])
    return vals[::-1] # exponentially weighted average for each suffix

@jax.jit
def compute_reward(prev_state, action, next_state):
    return 1
        
@jax.jit
def pi_loss(params, states, acts, adv): # pi, config.rollouts_per_epoch become compile time constants
    logits = pi.apply(params, states) # pi(s)
    log_probs = nn.log_softmax(logits) # pi(s) -> log p(*|s)
    log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze() # log p(*|s) -> log p(a|s)
    res = log_probs * adv # log p(a|s) -> adv * log p(a|s)
    return -1/config.rollouts_per_epoch * jnp.sum(res) # 1/|D| * sum(adv * log p(a|s))
    # take direction of gradient step to maximize loss

@jax.jit
def v_loss(params, states, rewards): # config.a traced during compile and becomes constant
    discount_rewards = suf_sum_decay(rewards, config.a) # unbiased estim for V(s)
    pred = V_net.apply(params, states).squeeze()
    return jnp.mean((pred - discount_rewards) ** 2) # L2 loss

@jax.jit
def update(params, grad): # TODO: Use an optimizer
    return jax.tree_util.tree_map(lambda u, v: u - config.lr * v, params, grad)

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
        
        logits = pi.apply(pi_params, S[-1]).squeeze() # (1, 2) output -> squeeze -> (2)
        key, subkey = jax.random.split(key, 2) # split key to sample action
        action = int(jax.random.categorical(key=subkey, logits=logits)) # jax.random.categorical directly samples from logits
        
        A.append(action)
        
        obs, _, terminated, truncated, info = env.step(action) # apply action, advance environment
        nxt_state = preprocess(env.render()), obs # get next state
        
        R.append(compute_reward(state, action, nxt_state)) # compute reward
        
        if terminated or truncated:
            break

        state = nxt_state

    return jnp.asarray(S), jnp.asarray(A), jnp.asarray(R)

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
    pi, V_net = NN(out_dim=2), NN(out_dim=1)

    pi_params = pi.init(key, jnp.zeros((1, config.img_size, config.img_size, config.n_frames)))
    v_params = V_net.init(key, jnp.zeros((1, config.img_size, config.img_size, config.n_frames)))
    
    pi_grad = jax.jit(jax.value_and_grad(pi_loss, argnums=0)) # d pi_loss / d pi_params
    v_grad = jax.jit(jax.value_and_grad(v_loss, argnums=0)) # d v_loss / d v_params
    
    for epoch in trange(1, config.epochs + 1):
        S, A, R = [], [], []
        
        # monte carlo estimation of raw rollouts
        key, *subkeys = jax.random.split(key, n_rollouts + 2)
        rollout(subkeys[-1], env) # for recording purposes
        for sample in trange(n_rollouts):
            episode = rollout(subkeys[sample], None)
            S.append(episode[0])
            A.append(episode[1])
            R.append(episode[2])
            
        S, A, R = jnp.concat(S, axis=0), jnp.concat(A, axis=0), jnp.concat(R, axis=0)
        
        # Calculate advantages
        V = V_net.apply(v_params, S).squeeze() # (B, ), where B = # steps
        Adv = comp_adv(R, V, config.a, config.b)
        
        # Calc policy grad & update policy
        prev_val, grad = pi_grad(pi_params, S, A, Adv) # computes 
        pi_params = update(pi_params, grad)
        
        
        flat_g, _ = ravel_pytree(grad)
        pi_norm = jnp.linalg.norm(flat_g)
        
        nxt_value, grad = pi_grad(pi_params, S, A, Adv) # check to make sure nxt_val - prev_val > 0
        pi_diff = nxt_value - prev_val
        
        # Calc v_net grad & update v_net
        prev_val, grad = v_grad(v_params, S, R)
        v_params = update(v_params, grad)
        
        
        flat_g, _ = ravel_pytree(grad)
        v_norm = jnp.linalg.norm(flat_g)
        
        nxt_val, grad = v_grad(v_params, S, R)
        v_diff = nxt_val - prev_val
        
        # logging
        stats = {
            "pi_diff": pi_diff,
            "pi_norm": pi_norm,
            "v_diff": v_diff,
            "v_norm": v_norm,
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
