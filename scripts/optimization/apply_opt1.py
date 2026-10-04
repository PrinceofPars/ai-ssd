import sys

path = "/home/ubuntu/ai-ssd/person1_kv_engine/real_llm/aissd_inference.py"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "# Attention computation on retrieved blocks" in line:
        new_lines.append("                # Attention computation on retrieved blocks (Optimized Native GQA SDPA)\n")
        new_lines.append("                t_attn = time.perf_counter()\n")
        new_lines.append("                scaling = attn_module.scaling\n")
        new_lines.append("                out = torch.nn.functional.scaled_dot_product_attention(\n")
        new_lines.append("                    q, act_k, act_v, scale=scaling, enable_gqa=True\n")
        new_lines.append("                )\n")
        new_lines.append("                out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()\n")
        new_lines.append("                model_timings[\"attn_matmul_s\"] += time.perf_counter() - t_attn\n\n")
        skip = True
        continue
    if skip:
        if "t_out = time.perf_counter()" in line:
            skip = False
            new_lines.append(line)
        continue
    new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)
print("Applied OPT-001 patch successfully")
