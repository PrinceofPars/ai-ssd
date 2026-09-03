"""
AI-SSD Interactive Dashboard (Streamlit + Matplotlib)
Visualizes KV cache savings, traffic reduction, FTL channel latency, and prefetch hit rates.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AI-SSD Simulator", layout="wide")

st.title("⚡ AI-SSD Co-Design Simulator")
st.markdown("Co-designing LLM KV Cache Offloading with Tensor-Aware Flash Storage Architecture")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
context_len = st.sidebar.selectbox("Context Length", [4096, 8192, 16384, 32768], index=3)
precision = st.sidebar.selectbox("KV Precision", ["FP16", "FP8"], index=0)
block_tokens = st.sidebar.selectbox("Block Size (Tokens)", [16, 32, 64], index=0)
offload_pct = st.sidebar.slider("KV Offload % to SSD", min_value=0, max_value=95, value=80, step=5)
topk_pct = st.sidebar.slider("Top-k Sparse Attention %", min_value=1, max_value=50, value=10, step=1)
ftl_mode = st.sidebar.radio("FTL Mapping Strategy", ["Tensor-Aware (Striped)", "Conventional (Sequential)"])
prefetch_enabled = st.sidebar.checkbox("Speculative Prefetching Enabled", value=True)

# Calculation Logic
bytes_per_elem = 2 if precision == "FP16" else 1
# 32 layers, 32 heads, 128 dim
total_kv_bytes = 2 * 32 * 32 * 128 * context_len * bytes_per_elem
total_kv_mb = total_kv_bytes / (1024 * 1024)

ram_saved_pct = offload_pct * 0.90  # GPU/RAM reduction
data_reduction_pct = 100.0 - topk_pct

# Latency calculations
base_latency = 100.0  # ms
ftl_speedup = 3.2 if "Tensor-Aware" in ftl_mode else 1.0
prefetch_bonus = 0.85 if prefetch_enabled else 1.0

est_latency = base_latency + (offload_pct * 0.3) / (ftl_speedup * (1.2 if prefetch_enabled else 1.0))
throughput = (1000.0 / est_latency) * (1.0 + (data_reduction_pct / 200.0))
prefetch_hit_rate = 88.0 if prefetch_enabled else 0.0

# Top Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("RAM Saved", f"{ram_saved_pct:.1f}%", f"{total_kv_mb * (ram_saved_pct/100):.0f} MB offloaded")
col2.metric("I/O Traffic Reduction", f"{data_reduction_pct:.1f}%", f"{100-topk_pct}% sparse")
col3.metric("Latency", f"{est_latency:.1f} ms", f"{'Fast' if 'Tensor-Aware' in ftl_mode else 'Slow'}")
col4.metric("Throughput", f"{throughput:.1f} tok/s")
col5.metric("Prefetch Hit Rate", f"{prefetch_hit_rate:.1f}%")

# Main Charts
st.subheader("Performance Breakdown")
c1, c2 = st.columns(2)

with c1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    tiers = ["GPU VRAM", "SSD Flash"]
    sizes = [total_kv_mb * (100 - offload_pct) / 100.0, total_kv_mb * offload_pct / 100.0]
    colors = ["#4285F4", "#EA4335"]
    ax1.bar(tiers, sizes, color=colors, width=0.5)
    ax1.set_ylabel("KV Cache Memory (MB)")
    ax1.set_title(f"Memory Distribution at {context_len} Context ({precision})")
    for i, v in enumerate(sizes):
        ax1.text(i, v + (max(sizes)*0.02), f"{v:.1f} MB", ha="center", fontweight="bold")
    st.pyplot(fig1)

with c2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    channels = [f"Ch {i}" for i in range(8)]
    if "Tensor-Aware" in ftl_mode:
        load = [12.5 + np.random.uniform(-1, 1) for _ in range(8)]
    else:
        load = [60.0, 25.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    ax2.bar(channels, load, color="#34A853")
    ax2.set_ylabel("Channel Utilization (%)")
    ax2.set_title("NAND Channel Bus Contention")
    ax2.set_ylim(0, 100)
    st.pyplot(fig2)

st.success("AI-SSD Architecture Ready: Person 1 (KV Engine) ↔ Person 2 (SSD Simulator) ↔ Person 3 (API Gateway)")
