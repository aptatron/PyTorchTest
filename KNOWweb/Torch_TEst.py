import torch
import torch.utils.benchmark as benchmark

# Define functions to benchmark
def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to `bmm`'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# Input for benchmarking
x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

# Benchmark with torch.utils.benchmark.Timer
t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    globals={'x': x, 'batched_dot_mul_sum': batched_dot_mul_sum}
)
t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    globals={'x': x, 'batched_dot_bmm': batched_dot_bmm}
)

print(t0.timeit(100))
print(t1.timeit(100))
