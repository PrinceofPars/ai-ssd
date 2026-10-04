#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/ai-ssd")
daemon_path = ROOT / "scripts" / "nvme_guest_daemon.c"
content = daemon_path.read_text()

# 1. Add TOPK_CHUNK_SIZE and buffer definition
old_buf_def = "    static char io_buf[BUFFER_SIZE];"
new_buf_def = """#define TOPK_CHUNK_SIZE (4 * 1024 * 1024) /* 4 MB coalesced read window */
    static char io_buf[BUFFER_SIZE];
    static char topk_chunk_buf[TOPK_CHUNK_SIZE];"""

assert old_buf_def in content, "Old buffer definition not found"
content = content.replace(old_buf_def, new_buf_def, 1)

# 2. Replace the candidate loop in OP_COMPUTE_TOPK
old_topk_loop = """                /* In-storage execution: read candidate K pages directly from NVMe and compute dot-product scores */
                for (uint32_t i = 0; i < num_cands; i++) {
                    uint32_t item_len = cands[i].length;
                    if (item_len > BUFFER_SIZE) item_len = BUFFER_SIZE;
                    ssize_t pr = pread(dev_fd, io_buf, item_len, (off_t)cands[i].offset);
                    if (pr != (ssize_t)item_len) continue;

                    float score = compute_block_score_gqa(
                        query_buf, (const float *)io_buf, (int)cands[i].actual_tokens,
                        (int)theader.q_heads, (int)theader.kv_heads, (int)theader.head_dim, theader.scale
                    );"""

new_topk_loop = """                /* In-storage execution: coalesced sliding window reads of candidate K pages directly from NVMe */
                uint64_t win_start = 0;
                uint64_t win_end = 0;
                int win_valid = 0;

                for (uint32_t i = 0; i < num_cands; i++) {
                    uint64_t cand_off = cands[i].offset;
                    uint32_t item_len = cands[i].length;

                    if (item_len > TOPK_CHUNK_SIZE) {
                        continue;
                    }

                    if (!win_valid || cand_off < win_start || (cand_off + item_len) > win_end) {
                        win_start = cand_off;
                        ssize_t pr = pread(dev_fd, topk_chunk_buf, TOPK_CHUNK_SIZE, (off_t)win_start);
                        if (pr <= 0) {
                            win_valid = 0;
                            continue;
                        }
                        win_end = win_start + (uint64_t)pr;
                        if ((cand_off + item_len) > win_end) {
                            win_valid = 0;
                            continue;
                        }
                        win_valid = 1;
                    }

                    const float *k_data = (const float *)(topk_chunk_buf + (cand_off - win_start));
                    float score = compute_block_score_gqa(
                        query_buf, k_data, (int)cands[i].actual_tokens,
                        (int)theader.q_heads, (int)theader.kv_heads, (int)theader.head_dim, theader.scale
                    );"""

assert old_topk_loop in content, "Old topk loop not found"
content = content.replace(old_topk_loop, new_topk_loop, 1)

daemon_path.write_text(content)
print("Updated scripts/nvme_guest_daemon.c with coalesced candidate reads.")
