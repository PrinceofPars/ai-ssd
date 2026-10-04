import time
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
hidden_size = 2560
q_dim = 2560
kv_dim = 256

x = torch.randn(1, 1, hidden_size)

w_q = torch.randn(q_dim, hidden_size)
w_k = torch.randn(kv_dim, hidden_size)
w_v = torch.randn(kv_dim, hidden_size)

w_qkv = torch.cat([w_q, w_k, w_v], dim=0).contiguous()

# Warmup
for _ in range(20):
    q = F.linear(x, w_q)
    k = F.linear(x, w_k)
    v = F.linear(x, w_v)

for _ in range(20):
    qkv = F.linear(x, w_qkv)
    q2 = qkv[..., :q_dim]
    k2 = qkv[..., q_dim:q_dim+kv_dim]
    v2 = qkv[..., q_dim+kv_dim:]

diff = (q - q2).abs().max() + (k - k2).abs().max() + (v - v2).abs().max()
print(f"Numerical difference: {diff.item():.2e}")

iters = 500
t0 = time.perf_counter()
for _ in range(iters):
    q = F.linear(x, w_q)
    k = F.linear(x, w_k)
    v = F.linear(x, w_v)
t_sep = (time.perf_counter() - t0) / iters

t0 = time.perf_counter()
for _ in range(iters):
    qkv = F.linear(x, w_qkv)
    q2 = qkv[..., :q_dim]
    k2 = qkv[..., q_dim:q_dim+kv_dim]
    v2 = qkv[..., q_dim+kv_dim:]
t_fused = (time.perf_counter() - t0) / iters

print(f"Separate Q, K, V: {t_sep*1000:6.3f} ms | 540 layers: {t_sep*540:6.3f} s")
print(f"Fused QKV:        {t_fused*1000:6.3f} ms | 540 layers: {t_fused*540:6.3f} s")
print(f"Speedup: {t_sep/t_fused:.2f}x | Potential savings: {(t_sep - t_fused)*540:.3f} s")
