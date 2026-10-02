#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# AI-SSD V2 SETUP
#
# Purpose:
#   Create the V2 branch, three agent worktrees, shared folders,
#   status/state files, and basic coordination infrastructure.
#
# This script DOES NOT define the daily work process.
# That will be designed separately.
# ============================================================

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

V2_BRANCH="v2-real-llm-kvssd"

P1_BRANCH="v2/p1-real-llm-kv"
P2_BRANCH="v2/p2-femu-ftl"
P3_BRANCH="v2/p3-system-integration"

P1_DIR="${PROJECT_ROOT}/../${PROJECT_NAME}-p1"
P2_DIR="${PROJECT_ROOT}/../${PROJECT_NAME}-p2"
P3_DIR="${PROJECT_ROOT}/../${PROJECT_NAME}-p3"

echo
echo "============================================================"
echo "              AI-SSD V2 SETUP"
echo "============================================================"
echo
echo "Project root : ${PROJECT_ROOT}"
echo "V2 branch   : ${V2_BRANCH}"
echo
echo "P1 worktree : ${P1_DIR}"
echo "P2 worktree : ${P2_DIR}"
echo "P3 worktree : ${P3_DIR}"
echo
echo "============================================================"
echo

# ------------------------------------------------------------
# 0. Verify repository
# ------------------------------------------------------------

if [ ! -d "${PROJECT_ROOT}/.git" ] && \
   [ ! -f "${PROJECT_ROOT}/.git" ]; then
    echo "[ERROR] This is not a git repository."
    exit 1
fi

echo "[1/8] Repository verified."

# ------------------------------------------------------------
# 1. Create V2 branch
# ------------------------------------------------------------

echo
echo "[2/8] Preparing V2 branch..."

CURRENT_BRANCH="$(git branch --show-current)"

if git show-ref --verify --quiet "refs/heads/${V2_BRANCH}"; then

    echo "V2 branch already exists."

else

    echo "Creating ${V2_BRANCH} from ${CURRENT_BRANCH}..."

    git branch "${V2_BRANCH}"

fi

# ------------------------------------------------------------
# 2. Checkout V2 in main worktree
# ------------------------------------------------------------

if [ "${CURRENT_BRANCH}" != "${V2_BRANCH}" ]; then

    echo "Switching main worktree to ${V2_BRANCH}..."

    git checkout "${V2_BRANCH}"

fi

# ------------------------------------------------------------
# 3. Create V2 directory structure
# ------------------------------------------------------------

echo
echo "[3/8] Creating V2 directory structure..."

mkdir -p \
    common/schemas \
    common/interfaces \
    common/experiment \
    docs/v2 \
    docs/v2/agents \
    docs/v2/research \
    docs/v2/proposals \
    docs/v2/handoffs \
    shared/traces \
    shared/traces/real_llm \
    shared/results \
    shared/results/ftl \
    shared/results/experiments \
    scripts

# Existing project ownership directories
mkdir -p \
    person1_kv_engine \
    person2_ssd \
    person3_system

# ------------------------------------------------------------
# 4. Create initial shared state
# ------------------------------------------------------------

echo
echo "[4/8] Creating shared V2 state files..."

cat > docs/v2/STATUS.md <<'EOF'
# AI-SSD V2 — GLOBAL STATUS

## Project

Real LLM + Real KV Cache + NVMe + FEMU + Tensor-Aware FTL

## Current Phase

SETUP COMPLETE

## Current Milestone

M0

## Last Global Update

Not started.

---

# Agent State

| Agent | Worktree | Branch | State |
|---|---|---|---|
| P1 | ../ai-ssd-p1 | v2/p1-real-llm-kv | NOT STARTED |
| P2 | ../ai-ssd-p2 | v2/p2-femu-ftl | NOT STARTED |
| P3 | ../ai-ssd-p3 | v2/p3-system-integration | NOT STARTED |

---

# Dependency State

No active dependencies.

---

# Integration State

Not started.

---

# Important

This file is shared state.

Agents must not overwrite another agent's status.

Each agent owns:

docs/v2/agents/P1_STATUS.md
docs/v2/agents/P2_STATUS.md
docs/v2/agents/P3_STATUS.md

---

# Next Step

The day-by-day implementation process and orchestration prompts
will be defined separately.
EOF


cat > docs/v2/ARCHITECTURE.md <<'EOF'
# AI-SSD V2 Architecture

## Target

Build a progressively realistic AI-SSD system around:

Real LLM
    ↓
Real KV Cache
    ↓
KV Management / Paging
    ↓
Storage Backend
    ↓
NVMe / FEMU
    ↓
FTL
    ↓
Tensor-Aware Placement
    ↓
Prefetching
    ↓
End-to-End Evaluation

## Main Workstreams

### P1

Real LLM
Real KV cache
Attention
Top-k KV retrieval
KV trace generation

### P2

FEMU
Virtual NVMe
FTL
NAND/channel model
Tensor-aware mapping

### P3

Storage abstraction
I/O
Prefetch
Experiment framework
Integration
Benchmarking

## Important

This document establishes the high-level architecture only.

The detailed execution schedule will be defined separately.
EOF


cat > docs/v2/CONTRACTS.md <<'EOF'
# AI-SSD V2 Contracts

This document will contain the shared interfaces between P1, P2
and P3.

The contracts should be frozen before major integration work.

Potential shared objects include:

- KVBlock
- KVRequest
- KVResponse
- KVTrace
- StorageBackend
- Metrics
- ExperimentConfig
- ExperimentResult

No agent should silently change a shared contract.

Contract changes should be documented in:

docs/v2/proposals/
EOF


cat > docs/v2/DECISIONS.md <<'EOF'
# AI-SSD V2 Decisions

This file records important project-wide technical decisions.

Format:

## DECISION-XXX

Date:

Decision:

Reason:

Alternatives considered:

Affected agents:

Status:
EOF


cat > docs/v2/ENVIRONMENT.md <<'EOF'
# AI-SSD V2 Environment

## Target Shared Machine

Maximum:

- 16 vCPU
- 64 GB RAM

## Architecture

Expected:

- x86_64
- Linux
- EC2

## Agent Resource Budget

Initial soft allocation:

P1:
8 vCPU / 32 GB

P2:
6 vCPU / 24 GB

P3:
2 vCPU / 8 GB

These are development guidelines, not hard CPU pinning.

## Environment Details

To be filled during environment validation.

CPU:
TBD

RAM:
TBD

Kernel:
TBD

OS:
TBD

Python:
TBD

Docker:
TBD

QEMU:
TBD

KVM:
TBD

FEMU:
TBD

fio:
TBD
EOF


cat > docs/v2/EXPERIMENT_PROTOCOL.md <<'EOF'
# AI-SSD V2 Experiment Protocol

To be defined during the implementation planning phase.

The eventual objective is to compare storage architectures while
keeping the workload and model conditions controlled.

Do not record final benchmark claims in this file yet.
EOF

# ------------------------------------------------------------
# 5. Create per-agent state files
# ------------------------------------------------------------

echo
echo "[5/8] Creating per-agent state files..."

create_agent_status() {

    local AGENT="$1"
    local ROLE="$2"
    local BRANCH="$3"
    local WORKTREE="$4"

    cat > "docs/v2/agents/${AGENT}_STATUS.md" <<EOF
# ${AGENT} — STATUS

## Identity

Agent:
${AGENT}

Role:
${ROLE}

Branch:
${BRANCH}

Worktree:
${WORKTREE}

---

## Current State

NOT STARTED

---

## Current Session

Session ID:
None

Started:
None

Last Updated:
None

---

## Current Milestone

None

---

## Current Task

None

---

## Completed

None

---

## Working On

None

---

## Next

None

---

## Blockers

None

---

## Dependencies

None

---

## Artifacts Created

None

---

## Tests

None

---

## Benchmarks

None

---

## Decisions

None

---

## Research

None

---

## Handoff Notes

None

---

## Last Commit

None

---

## Last Commit Message

None

---

# Usage / Session History

See:

docs/v2/agents/${AGENT}_USAGE.md
EOF
}


create_agent_status \
    "P1" \
    "Real LLM / KV / Attention / Top-k" \
    "${P1_BRANCH}" \
    "${P1_DIR}"


create_agent_status \
    "P2" \
    "FEMU / NVMe / FTL / NAND" \
    "${P2_BRANCH}" \
    "${P2_DIR}"


create_agent_status \
    "P3" \
    "Storage API / Prefetch / Experiments / Integration" \
    "${P3_BRANCH}" \
    "${P3_DIR}"

# ------------------------------------------------------------
# 6. Create usage tracking files
# ------------------------------------------------------------

echo
echo "[6/8] Creating usage/session tracking..."

create_usage_file() {

    local AGENT="$1"
    local ROLE="$2"

    cat > "docs/v2/agents/${AGENT}_USAGE.md" <<EOF
# ${AGENT} — USAGE / SESSION LOG

## Agent

${AGENT}

## Role

${ROLE}

---

# Current Session

Session:
None

Started:
None

Last Refresh:
None

Current State:
NOT STARTED

---

# How To Use This File

This file exists so the orchestration agent can recover state after
its usage limit or context/session refresh.

At the START of every session, record:

- session identifier
- timestamp
- current milestone
- current task

During the session, update important progress.

At the END of a session, record:

- completed work
- incomplete work
- exact next action
- files changed
- tests performed
- commit hash
- blockers
- dependencies
- anything the next session must know

---

# Session History

## Session 000

Status:
SETUP ONLY

Completed:
Repository V2 setup.

Remaining:
Start implementation planning.

Commit:
N/A

---

# Resource Notes

CPU:
TBD

RAM:
TBD

Disk:
TBD

Other:
TBD

---

# Usage Limit Notes

Record important usage/session observations here.

Do not record sensitive account information.
EOF
}


create_usage_file \
    "P1" \
    "Real LLM / KV / Attention / Top-k"


create_usage_file \
    "P2" \
    "FEMU / NVMe / FTL / NAND"


create_usage_file \
    "P3" \
    "Storage API / Prefetch / Experiments / Integration"

# ------------------------------------------------------------
# 7. Create shared handoff/proposal structure
# ------------------------------------------------------------

echo
echo "[7/8] Creating handoff and proposal templates..."

cat > docs/v2/handoffs/README.md <<'EOF'
# V2 Handoffs

Use this directory when one agent needs to leave a clear handoff
for another agent.

A handoff should contain:

- From
- To
- Date/session
- Completed work
- Required artifact
- Current state
- Exact next action
- Files involved
- Commit hash
- Known limitations
EOF


cat > docs/v2/proposals/README.md <<'EOF'
# V2 Contract Proposals

Do not directly modify shared contracts when the change affects
another agent.

Create a proposal containing:

- Current contract
- Proposed contract
- Reason
- Affected agents
- Migration requirements
- Compatibility considerations
EOF


cat > docs/v2/research/README.md <<'EOF'
# V2 Research

Research performed by agents should be recorded when it affects
an implementation decision.

Suggested format:

Question:

Source:

Finding:

Impact:

Decision:

Agent:
EOF

# ------------------------------------------------------------
# 8. Create worktrees
# ------------------------------------------------------------

echo
echo "[8/8] Creating agent worktrees..."

create_worktree() {

    local DIR="$1"
    local BRANCH="$2"

    if [ -e "$DIR" ]; then

        echo
        echo "Worktree path already exists:"
        echo "  $DIR"

        if git worktree list | grep -Fq "$DIR"; then
            echo "Already registered as a git worktree."
            return 0
        fi

        echo "[ERROR] Directory exists but is not a registered worktree."
        echo "Please resolve it manually before rerunning setup."
        exit 1
    fi

    if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then

        echo "Branch ${BRANCH} already exists."
        git worktree add "$DIR" "$BRANCH"

    else

        echo "Creating branch ${BRANCH}..."
        git worktree add -b "$BRANCH" "$DIR" "$V2_BRANCH"

    fi
}


create_worktree "$P1_DIR" "$P1_BRANCH"
create_worktree "$P2_DIR" "$P2_BRANCH"
create_worktree "$P3_DIR" "$P3_BRANCH"

# ------------------------------------------------------------
# Copy the setup state into each worktree
#
# Because worktrees share the same git object database, the files
# become visible after the setup commit below.
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Creating setup commit..."
echo "============================================================"

git add \
    common \
    docs/v2 \
    shared \
    scripts

git commit -m "v2: initialize parallel agent workspace" || true

# ------------------------------------------------------------
# Sync worktrees to the setup commit
# ------------------------------------------------------------

echo
echo "Syncing agent worktrees..."

git -C "$P1_DIR" checkout "$P1_BRANCH" >/dev/null 2>&1 || true
git -C "$P2_DIR" checkout "$P2_BRANCH" >/dev/null 2>&1 || true
git -C "$P3_DIR" checkout "$P3_BRANCH" >/dev/null 2>&1 || true

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

echo
echo "============================================================"
echo "                 SETUP COMPLETE"
echo "============================================================"
echo

echo "Main:"
echo "  ${PROJECT_ROOT}"
echo "  branch: ${V2_BRANCH}"
echo

echo "P1:"
echo "  ${P1_DIR}"
echo "  branch: ${P1_BRANCH}"
echo

echo "P2:"
echo "  ${P2_DIR}"
echo "  branch: ${P2_BRANCH}"
echo

echo "P3:"
echo "  ${P3_DIR}"
echo "  branch: ${P3_BRANCH}"
echo

echo "Git worktrees:"
git worktree list

echo
echo "V2 files:"
echo
echo "  docs/v2/STATUS.md"
echo "  docs/v2/ARCHITECTURE.md"
echo "  docs/v2/CONTRACTS.md"
echo "  docs/v2/DECISIONS.md"
echo "  docs/v2/ENVIRONMENT.md"
echo "  docs/v2/EXPERIMENT_PROTOCOL.md"
echo
echo "Agent state:"
echo
echo "  docs/v2/agents/P1_STATUS.md"
echo "  docs/v2/agents/P1_USAGE.md"
echo "  docs/v2/agents/P2_STATUS.md"
echo "  docs/v2/agents/P2_USAGE.md"
echo "  docs/v2/agents/P3_STATUS.md"
echo "  docs/v2/agents/P3_USAGE.md"
echo
echo "============================================================"
echo "Next step:"
echo "Design the day-by-day execution process and agent prompts."
echo "============================================================"
echo