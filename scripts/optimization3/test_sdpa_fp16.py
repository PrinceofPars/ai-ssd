import torch

# Test SDPA with GQA in FP16 on CPU
q = torch.randn(1, 32, 1, 128, dtype=torch.float16)
k = torch.randn(1, 8, 276, 128, dtype=torch.float16)
v = torch.randn(1, 8, 276, 128, dtype=torch.float16)

try:
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / (128 ** 0.5), enable_gqa=True)
    print("SDPA GQA FP16 succeeded! Output shape:", out.shape, "dtype:", out.dtype)
except Exception as e:
    print("SDPA GQA FP16 failed:", e)
    # Test fallback: expand k and v
    gqa_ratio = 32 // 8
    k_exp = k.repeat_interleave(gqa_ratio, dim=1)
    v_exp = v.repeat_interleave(gqa_ratio, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, scale=1.0 / (128 ** 0.5))
    print("SDPA expanded succeeded! Output shape:", out.shape, "dtype:", out.dtype)
