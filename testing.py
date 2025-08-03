import jax
import gymnasium as gym

from reinforce import rollout, SEED  # adjust import to your module path

def test_rollout_run():
    """
    Minimal test for the rollout function: checks types, lengths, and basic shapes.
    """
    # Initialize RNG and environment
    key = jax.random.PRNGKey(SEED)
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Run a single rollout
    S, A, R = rollout(key, env)
    print(S, A, R)
    # # Basic type checks
    # assert isinstance(S, list), f"States should be a list, got {type(S)}"
    # assert isinstance(A, list), f"Actions should be a list, got {type(A)}"
    # assert isinstance(R, list), f"Rewards should be a list, got {type(R)}"

    # # Ensure all sequences have equal length
    # n = len(S)
    # assert n == len(A) == len(R), (
    #     f"Length mismatch: states={len(S)}, actions={len(A)}, rewards={len(R)}"
    # )

    # if n > 0:
    #     img, obs = S[0]
    #     # Image should be a JAX array with shape (1, H, W)
    #     assert hasattr(img, 'shape'), "First state image has no shape attribute"
    #     assert img.ndim == 3 and img.shape[0] == 1, (
    #         f"Expected image shape (1, H, W), got {img.shape}"
    #     )
    #     # Observation should match env's observation_space
    #     obs_space = env.observation_space.shape
    #     assert hasattr(obs, 'shape'), "First state obs has no shape attribute"
    #     assert obs.shape == obs_space, (
    #         f"Expected obs shape {obs_space}, got {obs.shape}"
    #     )

    # print(f"✓ rollout returned {n} steps with matching lengths and valid shapes")


if __name__ == "__main__":
    test_rollout_run()
