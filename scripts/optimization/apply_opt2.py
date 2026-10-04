#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/ai-ssd")

client_path = ROOT / "person2_ssd" / "nvme_client.py"
c_content = client_path.read_text()

old_unpack = 'magic, status, op, _, _, resp_items = struct.unpack("<IBBHQI", resp)'
new_unpack = 'magic, status, op, _, resp_offset, resp_items = struct.unpack("<IBBHQI", resp)'

assert old_unpack in c_content, "Old unpack not found"
c_content = c_content.replace(old_unpack, new_unpack)
client_path.write_text(c_content)
print("Fixed nvme_client.py unpack.")
