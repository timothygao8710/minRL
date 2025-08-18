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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

'''
First, let's define fQ = r(s, a, s') + theta max_a' Q(s', a')

Second, recall the essential idea of Q-learning is to indirectly learn the optimal policy by learning the Q-function it would have.

Suppose we have some optimal pi, then apparently Q = fQ. In other words, Q = fQ is nessesary for an optimal pi, 
(pi is optimal) => Q = fQ

Q-learning makes 2 assumptions:
1. Q = fQ is also sufficient for an optimal pi. (Q = fQ) => (the pi described by Q is optimal)
2. The distance |Q - fQ| is a measurement for how far Q is from describing the optimal policy

We train on (Q - fQ)^2 to minimize this distance.
- One perspective on this training is DP, analogous to the relaxation step in e.g., Bellman Ford. Instead of shortest path, we have the 
maximum weight path, and DP(s, a) = maximum reward path taking edge a from state s ending at a terminating state; all initialized to -infinity.

- Another perspective is (NN captures behavior of optimal agent) => (loss is minimized) and we hope the converse is true. We hope that through
minimizing loss and seeing many examples, NN learns to characterize the optimal policy that we have in mind.

    - I don't believe what NN learns is really anything like the optimal policy that we have in mind. I believe learning what is optimal also requires examples of what is not, similar to policy gradient.

- A third perspective ... 

In the following code, max_a Q(s, a) is computed by iterating over action space
'''

action_space = jnp.array([0, 1])

wandb.init(
    entity="timothy-gao",
    project="RL",
    tags=["cartpole_no_img", "Q"],
    mode="offline", 
    # dir="/home/timothygao/SpinningUp/plots",
    config={
        "epochs": 100000,
        "buffer_size": 4096,
        # "buffer_size": 64,
        "steps_per_epoch": 200,
        "batch_size" : 0, # buffer size can't fit in one go

        "lr": 1e-3,
        "a": 0.99, # discount on rewards
        "n_frames": 2,
        "epoch_record_freq": 100,
        "seed": 42,
        "max_steps": 200,
        
        "end_val": 0 # value of a terminating state TODO: If truncated, end_val should be calced from last state
    }
)

config = wandb.config

class NN(nn.Module):
    
    @nn.compact
    def __call__(self, S, A):   
        # add batch dims
        if A.ndim == 0:
            A = A[None, None] # () -> (1, 1)
            
        if S.ndim == 1: # (N, ) => (1, N)
            S = S[None, :]

        if A.ndim == 1: # (L, ) => (L, 1)
            A = A[:, None]
        
        x = jnp.concat([S, A], axis = 1)
        
        for h in (32, 16):
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        
        return nn.Dense(1)(x)

@jax.jit
def compute_fQ(params, states, actions, rewards):
    L = len(states) # number of steps

    # compute max_a' Q(s', a')
    Q_actionspace = jax.lax.map(lambda action : 
        Q.apply(params, states, jnp.broadcast_to(action, (L, 1))).squeeze()
    , action_space) # 2, L
    
    Q_max = jnp.max(Q_actionspace, axis=0)[:, None] # L, 1
    
    Q_max = jnp.concat([Q_max[1:], jnp.array([[config.end_val]])], axis=0) # Q(terminating state, *) = 0
    
    # compute fQ = rewards + a * max_a' Q(s', a')
    fQ = rewards + config.a * Q_max
    return fQ # returns shape L, 1

@jax.jit
def q_loss(params, S, A, fQ):
    return jnp.mean((Q.apply(params, S, A) - fQ) ** 2)

def get_logging_info(params, S, A, fQ):
    pred_q = Q.apply(params, S, A)
    
    dists = fQ - pred_q # (N, 1)
    
    return {
        "dists_std" : jnp.std(dists),
        "dists_mean" : jnp.mean(dists),
        "dists_median" : jnp.median(dists),
        "dists_min": jnp.min(dists),
        "dists_max": jnp.max(dists),
        "dists_abs_mean": jnp.mean(jnp.abs(dists)),  # sqrt of loss
        "dists_percentile_25": jnp.percentile(dists, 25),
        "dists_percentile_75": jnp.percentile(dists, 75),
        "pred_q_mean": jnp.mean(pred_q),
        "pred_q_std": jnp.std(pred_q),
        "pred_q_min": jnp.min(pred_q),
        "pred_q_max": jnp.max(pred_q),
        "target_q_mean": jnp.mean(fQ),
        "target_q_std": jnp.std(fQ),
        "fraction_overestimate": jnp.mean(dists > 0),  # How often we overestimate
        "fraction_underestimate": jnp.mean(dists < 0),
        "target_pred_correlation": jnp.corrcoef(fQ.flatten(), pred_q.flatten())[0,1],
    }

@jax.jit
def update(params, grad, lr): # TODO: Use an optimizer
    return jax.tree_util.tree_map(lambda u, v: u - lr * v, params, grad)

@jax.jit
def rollout_helper(key, params): 
    env, env_params = gymnax.make("CartPole-v1")
    
    state, env_state = env.reset(key)

    def step(carry_i, _):  
        cur_key, state, env_state, prev_done = carry_i
              
        logits = jax.lax.map(lambda action : 
            Q.apply(params, state, action).squeeze()
        , action_space) # (1, 2)
        logits = logits.squeeze()
        
        cur_key, subkey = jax.random.split(cur_key, 2) # split key to sample action
        action = jax.random.categorical(key=subkey, logits=logits).squeeze()
            
        cur_key, subkey = jax.random.split(cur_key, 2) # split key to step env
        nxt_state, env_state, reward, done, info = env.step(subkey, env_state, action, env_params)
        
        done = jnp.logical_or(done, prev_done)
                    
        return (cur_key, nxt_state, env_state, done), (state, action, reward, done)

    _, (S, A, R, D) = jax.lax.scan(step, (key, state, env_state, False), xs=None, length=config.max_steps)
    
    return (S, A, R, D)

def rollout(key, params): 
    S, A, R, D = rollout_helper(key, params)
    
    # print(S.shape, A.shape, R.shape, D.shape)
    
    num_steps = jnp.sum(D)
    S, A, R = S[:num_steps], A[:num_steps][:, None], R[:num_steps][:, None]
    
    # print("input to fQ", S.shape, A.shape, R.shape)
    
    fQ = compute_fQ(params, S, A, R)
    
    # print("rollout return", S.shape, A.shape, R.shape, fQ.shape)
    
    return (S, A, fQ) # (L, 4) (L, 1) (L, 1)

def visualize_rollout(key, params, name): # visualizer a single rollout
    env, env_params = gymnax.make("CartPole-v1")
    
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)
        
        logits = jax.lax.map(lambda action : 
            Q.apply(params, states, action).squeeze()
        , action_space) # (1, 2)
        logits = logits.squeeze()
        
        action = jax.random.categorical(key=key_act, logits=logits).squeeze() # sampling using Q-value as logits
        
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
    key = jax.random.key(config.seed)
    Q = NN()

    params = Q.init(key, jnp.zeros((1, 4)), jnp.zeros((1, 1)))
    
    q_grad = jax.jit(jax.value_and_grad(q_loss, argnums=0)) # d q_loss / d q_params
        
    S_buffer, A_buffer, fQ_buffer = jnp.zeros((0, 4)), jnp.zeros((0, 1)), jnp.zeros((0, 1)) 
        
    episode_num = 0
    
    for epoch in trange(1, config.epochs + 1):
        key, subkey = jax.random.split(key)
        
        if epoch % 37 == 0:
            visualize_rollout(subkey, params, f"q_{epoch}_no_img")  # for recording purposes
        
        S_buffer, A_buffer, fQ_buffer = map(lambda x : x[config.steps_per_epoch:], [S_buffer, A_buffer, fQ_buffer])
        
        alive_times = [] # for logging
        
        while len(S_buffer) < config.buffer_size:
            key, subkey = jax.random.split(key)        
            S, A, fQ = rollout(subkey, params)
            
            S_buffer = jnp.concat([S_buffer, S], axis=0)
            A_buffer = jnp.concat([A_buffer, A], axis=0)
            fQ_buffer = jnp.concat([fQ_buffer, fQ], axis=0)
                        
            alive_times.append(len(S))
            episode_num += 1
        
        loss, grad = q_grad(params, S_buffer, A_buffer, fQ_buffer) # computes 
        params = update(params, grad, config.lr)
        
        stats = {
            "mean_alive_time": sum(alive_times) / len(alive_times), # TODO: Correlation between loss and mean_alive_time overtime
            "q_loss" : loss,
            
            "left_percentage" : jnp.sum(A_buffer == 0) / len(A),
            "epoch": epoch,
            "episode": episode_num,
        }
        
        stats = stats | get_logging_info(params, S_buffer, A_buffer, fQ_buffer)
        
        wandb.log(stats, step=episode_num)

    wandb.finish()
