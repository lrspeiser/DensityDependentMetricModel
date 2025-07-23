import jax
import jax.numpy as jnp
import time

def test_large_sum():
    x = jnp.arange(10**8, dtype=jnp.float32)

    start = time.time()
    result = jnp.sum(x).block_until_ready()
    end = time.time()

    print("Sum:", result)
    print(f"Time: {end - start:.3f} seconds")
    print(f"Backend: {jax.default_backend()}")

test_large_sum()
