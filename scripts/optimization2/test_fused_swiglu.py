import time
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
hidden_size = 2560
intermediate_size = 9728

x = torch.randn(1, 1, hidden_size)

w_gate = torch.randn(intermediate_size, hidden_size)
w_up = torch.randn(intermediate_size, hidden_size)
w_down = torch.randn(hidden_size, intermediate_size)

w_gate_up = torch.cat([w_gate, w_up], dim=0).contiguous()

# Warmup
for _ in range(10):
    g = F.linear(x, w_gate)
    u = F.linear(x, w_up)
    h = F.silu(g) * u
    out = F.linear(h, w_down)

for _ in range(10):
    gu = F.linear(x, w_gate_up)
    g2 = gu[..., :intermediate_size]
    u2 = gu[..., intermediate_size:]
    h2 = F.silu(g2) * u2
    out2 = F.linear(h2, w_down)

diff = (out - out2).abs().max()
print(f"Numerical diff: {diff.item():.2e}")

iters = 100
t0 = time.perf_counter()
for _ in range(iters):
    g = F.linear(x, w_gate)
    u = F.linear(x, w_up)
    h = F.silu(g) * u
    out = F.linear(h, w_down)
t_sep = (time.perf_counter() - t0) / iters

t0 = time.perf_counter()
for _ in range(iters):
    gu = F.linear(x, w_gate_up)
    g2 = gu[..., :intermediate_size]
    u2 = gu[..., intermediate_size:]
    h2 = F.silu(g2) * u2
    out2 = F.linear(h2, w_down)
t_fused = (time.perf_counter() - t0) / iters

print(f"Separate SwiGLU: {t_sep*1000:6.3f} ms | 540 layers: {t_sep*540:6.3f} s")
print(f"Fused Gate-Up:   {t_fused*1000:6.3f} ms | 540 layers: {t_fused*540:6.3f} s")
print(f"Speedup: {t_sep/t_fused:.2f}x | Potential savings: {(t_sep - t_fused)*540:.3f} s")
