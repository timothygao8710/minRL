
from jax.flatten_util import ravel_pytree
import wandb
import os, operator
import numpy as np

import jax, jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn
from tqdm import trange
from multiprocessing.pool import Pool
from PIL import Image
import gymnax
from gymnax.visualize import Visualizer

os.environ["SDL_AUDIODRIVER"] = "headless"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

'''
Solving the CartPole-v1 with only image observations

The main challenge is low signal to noise ratio

Builds on 1_reinforce with:
- Use value function for baseline (optimal control variate)
- GAE (higher bias for lower variance) 

For unbiased V, the TD residual a * V(s_i+1) + r_i - V(s_i) is unbiased estimator for a-just advantage (see paper)

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
    tags=["cartpole_no_img"],
    # mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
      "epochs": 10000,
      "rollouts_per_epoch": 256,
      "max_steps": 50,
      "lr": 1e-3,
      "a": 0.99, # discount on rewards
      "b": 0.98, # discount on TD residual, used in GAE
      "n_frames": 2,
      "epoch_record_freq": 100,
      "seed": 42,
      "img_size": 64,
      "end_val": 0 # value of a terminating state TODO: If truncated, end_val should be bootstrapped
    }
)

config = wandb.config

class NN(nn.Module):
    out_dim: int
    latent_dim: int = 64
    
    @nn.compact
    def __call__(self, S):        
        if len(S.shape) == 1:
            S = S[None, :]

        for h in (self.latent_dim, 16):
            S = nn.Dense(h)(S)
            S = nn.relu(S)
        
        return nn.Dense(self.out_dim)(S)

@jax.jit
def preprocess(img, out=config.img_size):
    img = jimage.resize(img, (out, out, 1), method='nearest')
    img = (img - 127.5) / 255.0
    return img

@jax.jit # get suffix sum with decay
def suf_sum_decay(vals, decay_coef): 
    
    def roll(carry_i, input_i): # (carry_i, input_i) -> (carry_i+1, output_i)
        return input_i + decay_coef * carry_i, input_i + decay_coef * carry_i
    
    _, vals = jax.lax.scan(roll, 0.0, vals[::-1])
    return vals[::-1] # exponentially weighted average for each suffix

@jax.jit
def comp_adv(R, V, a, b): # GAE https://arxiv.org/abs/1506.02438
    # comp td-resids
    # resids = R - V + a * jnp.concat([V[1:], jnp.array([config.end_val])])
    # return suf_sum_decay(resids, a * b)
    
    temp = suf_sum_decay(R, a)
    temp = jnp.mean(temp)
    return temp

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
def update(params, grad, lr): # TODO: Use an optimizer
    return jax.tree_util.tree_map(lambda u, v: u - lr * v, params, grad)

def hist(jnp_arr, name):
    import matplotlib.pyplot as plt
    plt.hist(np.array(jnp_arr).flatten())
    plt.savefig(f'/home/timothygao/minRL/images/{name}_distribution.png', dpi=300, bbox_inches='tight')  # Save first
    plt.clf() 

def rollout(key, pi_params): # returns a single traj = [states, actions, rewards]
    env, env_params = gymnax.make("CartPole-v1")
    
    state, env_state = env.reset(key)

    def step(carry_i, _):  
        cur_key, state, env_state, prev_done = carry_i
              
        logits = pi.apply(pi_params, state).squeeze()
        
        cur_key, subkey = jax.random.split(cur_key, 2) # split key to sample action
        action = jax.random.categorical(key=subkey, logits=logits).squeeze()
            
        cur_key, subkey = jax.random.split(cur_key, 2) # split key to step env
        nxt_state, env_state, reward, done, info = env.step(subkey, env_state, action, env_params)
        
        done = jnp.logical_or(done, prev_done)
                    
        return (cur_key, nxt_state, env_state, done), (state, action, reward, done)

    _, (S, A, R, D) = jax.lax.scan(step, (key, state, env_state, False), xs=None, length=500)
    
    return (S, A, R, D)

def visualize_rollout(key, pi_params, name): # visualizer a single rollout
    env, env_params = gymnax.make("CartPole-v1")
    
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)
        
        logits = pi.apply(pi_params, obs).squeeze()
        action = jax.random.categorical(key=key_act, logits=logits).squeeze()
        
        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"/home/timothygao/minRL/videos/{name}.gif")

if __name__ == '__main__':
    n_rollouts = config.rollouts_per_epoch
    
    key = jax.random.key(config.seed)
    pi, V_net = NN(out_dim=2), NN(out_dim=1)

    pi_params = pi.init(key, jnp.zeros((1, 4)))
    v_params = V_net.init(key, jnp.zeros((1, 4)))
    
    pi_grad = jax.jit(jax.value_and_grad(pi_loss, argnums=0)) # d pi_loss / d pi_params
    v_grad = jax.jit(jax.value_and_grad(v_loss, argnums=0)) # d v_loss / d v_params
    
    lr = config.lr
    
    # rollout_all = jax.vmap(rollout, in_axes=(0, None))
    rollout_all = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    
    for epoch in trange(1, config.epochs + 1):
        key, subkey = jax.random.split(key)
        visualize_rollout(subkey, pi_params, f"{epoch}_no_img")
        
        # monte carlo estimation of raw rollouts
        key, *subkeys = jax.random.split(key, n_rollouts + 1)
        subkeys = jnp.stack(subkeys)
        
        S, A, R, D = rollout_all(subkeys, pi_params) # for recording purposes
                
        S, A, R, D = map(lambda x : jnp.stack(x, axis=0), [S, A, R, D])     
           
        S, A, R = S[~D], A[~D], R[~D]
                
        # Calculate advantages
        V = V_net.apply(v_params, S).squeeze() # (B, ), where B = # steps
        Adv = comp_adv(R, V, config.a, config.b)
        
        # Adv /= jnp.std(Adv)
        
        # Plot advantages and values
        
        # hist(Adv, f"{epoch}_Adv")
        # hist(V, f"{epoch}_V")
        # hist(suf_sum_decay(R, config.a), f"{epoch}_R")

        # Calc policy grad & update policy
        prev_val, grad = pi_grad(pi_params, S, A, Adv) # computes 
        pi_params = update(pi_params, grad, lr)
        
        # if v_loss_val.astype(int) < 1:
        #     pi_params = update(pi_params, grad, lr)
        
        # flat_g, _ = ravel_pytree(grad)
        # pi_norm = jnp.linalg.norm(flat_g)
        
        # nxt_value, grad = pi_grad(pi_params, S, A, Adv) # check to make sure nxt_val - prev_val > 0
        
        # pi_diff = nxt_value - prev_val
        
        # Calc v_net grad & update v_net
        v_loss_val, grad = v_grad(v_params, S, R)
        v_params = update(v_params, grad, lr)
        
        # flat_g, _ = ravel_pytree(grad)
        # v_norm = jnp.linalg.norm(flat_g)
        
        # nxt_val, grad = v_grad(v_params, S, R)
        # v_diff = nxt_val - prev_val
        
        # print(pi_diff, pi_norm, v_diff, v_norm)
                
        # logging
        R = suf_sum_decay(R, config.a)
        stats = {
            # "pi_diff": pi_diff,
            # "pi_norm": pi_norm,
            # "v_diff": v_diff,
            # "v_norm": v_norm,
            "mean_alive_time": len(S) / n_rollouts,
            "v_loss" : v_loss_val,
            
            "mean_adv": jnp.mean(Adv),
            "med_adv": jnp.median(Adv),
            "var_adv": jnp.var(Adv),
            "std_adv": jnp.std(Adv),
            
            "mean_reward": jnp.mean(R),
            "var_reward": jnp.var(R),
            "std_reward": jnp.std(R),
            
            "left_percentage" : jnp.sum(A == 0) / len(A),
            "learning_rate":lr,
            
            "epoch": epoch,
            "episode": epoch * n_rollouts,
        }
        
        wandb.log(stats, step=epoch * n_rollouts)

    env.close()
    wandb.finish()
