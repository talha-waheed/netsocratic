# NetSocratic

NetSocratic is a multi-agent LLM pipeline that translates vague network operator intent into precise, verified OSPF router configurations. It uses a Socratic clarification loop to resolve ambiguity, generates candidate configurations via an LLM, and uses [Batfish](https://www.batfish.org/) to formally verify and select among them.

---

## Architecture

The pipeline runs three agents in sequence, with an automatic recovery loop when selection fails.

```
Operator intent (vague)
        │
        ▼
┌─────────────────────┐
│  Clarification Agent │  ← asks targeted Q&A to remove ambiguity
└─────────────────────┘
        │ clarified intent
        ▼
┌─────────────────────┐
│   Generator Agent   │  ← produces N candidate OSPF configs (varied temperatures)
└─────────────────────┘
        │ N candidate dicts
        ▼
┌─────────────────────┐
│   Selection Agent   │  ← Batfish EC-pruning + distinguishing Q&A → 1 winner
└─────────────────────┘
        │ winner config (or recovery)
        ▼
  Network Specification
  results/<timestamp>/selection/winner/
```

If the Selection Agent cannot converge (operator rejects all options, or max rounds reached), it packages all state into a `RuntimeContext` and restarts the pipeline from the Clarification Agent with that context injected.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | Uses `str \| None` union syntax |
| OpenAI API key | — | For GPT-4o (or any `gpt-*` model) |
| [Batfish](https://batfish.readthedocs.io/en/latest/getting_started.html) | latest | Required for Selection Agent; not needed for `--dry-run` or `--skip-selector` |
| pybatfish | latest | `pip install pybatfish` |

Batfish must be running locally (default: `localhost:9996`) before invoking the full pipeline. See [Batfish quickstart](https://batfish.readthedocs.io/en/latest/getting_started.html) for Docker setup instructions.

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url> && cd netsocratic

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and TOPO_DIR (see Configuration below)
```

---

## Configuration

All settings are read from `.env` (via `python-dotenv`) or environment variables. Copy `.env.example` to `.env` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model name passed to the Chat Completions API |
| `MAX_CLARIFY_ROUNDS` | `5` | Maximum clarification rounds before best-effort synthesis |
| `NUM_CANDIDATES` | `3` | Number of candidate configurations the Generator produces |
| `KB_DIR` | `agents/knowledge-base` | Directory for knowledge-base Markdown/text files |
| `TOPO_DIR` | `../cs598_LMZ_final/topo` | Directory of base `.cfg` files — one per router |
| `RESULTS_DIR` | `results` | Root directory for all run output |

All `.env` values can be overridden at runtime with the corresponding CLI flag.

---

## Usage

```bash
python main.py --intent "Ensure connectivity between Athens and our servers in London through Paris"
```

### Options

```
--intent TEXT           Vague network intent as a string
--intent-file PATH      Path to a text file containing the intent

--model MODEL           OpenAI model (default: gpt-4o)
--max-rounds N          Max clarification rounds per pass (default: 5)
--num-candidates N      Candidate configs to generate (default: 3)
--kb-dir PATH           Knowledge-base directory (default: agents/knowledge-base)
--topo-dir PATH         Base topology directory with one .cfg per router
--results-dir PATH      Root output directory (default: results)
--batfish-script-dir P  Directory containing Batfish diff scripts (default: batfish)
--max-recovery-rounds N Max full-pipeline restarts on failure (default: 2)

--skip-generator        Run clarification only; stop before generation
--skip-selector         Run clarification + generation only; stop before selection
--dry-run               Skip all LLM and Batfish calls; use canned responses for testing
```

### Examples

```bash
# Full pipeline (requires Batfish running)
python main.py --intent "Athens must reach 100.0.29.0/24 via London, 3 ECMP paths"

# Clarification only (no LLM config generation)
python main.py --intent "..." --skip-generator

# Clarification + generation, no Batfish verification
python main.py --intent "..." --skip-selector

# Test the full pipeline without an API key or Batfish
python main.py --intent "..." --dry-run

# Use a different model with more candidates
python main.py --intent "..." --model gpt-4o-mini --num-candidates 5

# Load intent from a file
python main.py --intent-file intents/my_intent.txt
```

---

## Pipeline Details

### Phase 1 — Clarification Agent

The Clarification Agent removes ambiguity from the operator's intent through a structured Q&A loop. It uses two LLM passes per round:

1. **Question generation** (`CLARIFY_SYSTEM`) — identifies missing or underspecified information and asks targeted follow-up questions.
2. **Sufficiency check** (`SUFFICIENCY_SYSTEM`) — evaluates whether the accumulated Q&A is now sufficient to act on.

The agent targets three specific dimensions that the Generator Agent requires:

- **Reachability** — exact source node and destination subnet (CIDR)
- **Waypointing** — mandatory vs. preferred, and exact waypoint router name
- **Load balancing** — exact integer number of equal-cost (ECMP) paths

The loop runs until the LLM declares the intent fully specified, or `--max-rounds` is reached (best-effort synthesis in that case).

If a `RuntimeContext` from a previous failed attempt is provided, it is injected into every LLM prompt so the model knows what was already tried.

### Phase 2 — Generator Agent

The Generator Agent produces `N` candidate Cisco IOS OSPF configurations by calling the LLM with varied temperatures (default: `[0.2, 0.7, 1.0]`).

Key design decisions:
- **Single source of truth**: the network topology is parsed directly from the `TOPO_DIR` `.cfg` files at runtime (`generate_topology_doc()`). There is no static `topology.md`.
- The LLM is told to output **only the routers whose `ip ospf cost` values must change**. All other routers are copied verbatim from the base topology.
- Routing policy is implemented through cost values only: cost `1` = preferred path, cost `1000` = effectively blocked (mandatory waypointing).
- ECMP is achieved by making exactly N paths share the minimum total metric.

Output per candidate:
```
results/<timestamp>/candidates/candidate_N/
├── configs/
│   ├── Athens.cfg
│   ├── London.cfg
│   └── ...          (all routers — modified + base copies)
└── decision_summary.txt
```

### Phase 3 — Selection Agent

The Selection Agent narrows N candidates to a single Network Specification using two sub-steps:

#### EC-Based Pruning (Batfish)

All pairwise candidate comparisons are run via subprocess calls to the Batfish scripts:

- `batfish/diff_analysis.py` — **reachability** differences via `differentialReachability()`
- `batfish/diff_advanced.py` — **waypointing + load-balancing** differences via `reachability()` trace comparison

A union-find algorithm groups behaviorally identical candidates into equivalence classes. One representative per class survives.

#### Distinguishing Q&A Loop

For each surviving pair, the LLM converts the Batfish-detected behavioral difference into a single plain-English question for the operator. The operator's answer prunes candidates whose behavior matches the rejected side (using already-computed pairwise diffs — no Batfish re-runs).

Priority order for which difference to surface: **reachability > waypointing > load-balancing**.

#### Recovery

Two conditions trigger recovery:

1. **User rejection** — answer contains signals like `"none of these"`, `"try again"`, `"start over"`, `"wrong"`, etc.
2. **Max rounds exhausted** — `--max-recovery-rounds` full-pipeline restarts without converging.

On recovery, all state is packaged into a `RuntimeContext` and written to `results/<timestamp>/selection/runtime_context.txt`. The main loop re-runs the full pipeline with this context injected into the next Clarification pass.

---

## Output Structure

Each pipeline run produces a timestamped directory under `results/`:

```
results/2026-04-22_03-36-57/
├── intent_original.txt          # raw intent as provided
├── clarify_prompt.txt           # clarification system + user prompt
├── questions_round_1.txt        # questions asked in round 1
├── answers_round_1.txt          # operator's answers for round 1
├── clarified_intent.txt         # final output of Clarification Agent
├── clarify_runtime_context.txt  # (if recovery) injected context from prior run
│
├── candidates/
│   ├── candidate_1/
│   │   ├── configs/
│   │   │   ├── Athens.cfg
│   │   │   ├── London.cfg
│   │   │   └── ...
│   │   └── decision_summary.txt
│   ├── candidate_2/
│   │   └── ...
│   └── candidate_3/
│       └── ...
│
└── selection/
    ├── batfish/
    │   ├── candidate_1_vs_candidate_2_reachability.txt
    │   ├── candidate_1_vs_candidate_2_advanced.txt
    │   ├── candidate_1_vs_candidate_3_reachability.txt
    │   ├── candidate_1_vs_candidate_3_advanced.txt
    │   ├── candidate_2_vs_candidate_3_reachability.txt
    │   └── candidate_2_vs_candidate_3_advanced.txt
    ├── selection_log.txt          # EC grouping, Q&A rounds, pruning decisions
    ├── runtime_context.txt        # (if recovery) packed state for next Clarification pass
    └── winner/
        ├── decision_summary.txt
        └── configs/
            ├── Athens.cfg
            ├── London.cfg
            └── ...                # all router configs for the winning candidate
```

---

## Project Structure

```
netsocratic/
├── main.py                        # CLI entry point; orchestrates the full pipeline
├── config.py                      # Loads .env into named constants
├── requirements.txt
│
├── agents/
│   ├── clarification_agent.py     # Phase 1: Socratic Q&A to resolve intent ambiguity
│   ├── generator_agent.py         # Phase 2: multi-temperature OSPF config generation
│   ├── selection_agent.py         # Phase 3: Batfish EC-pruning + distinguishing Q&A
│   └── knowledge-base/
│       └── routing_policy.md      # Waypointing and ECMP policy guide (injected into generator)
│
├── batfish/
│   ├── diff_analysis.py           # Reachability diff via differentialReachability()
│   ├── diff_advanced.py           # Waypointing + load-balancing diff via reachability()
│   └── run_all_diffs.py           # Standalone batch runner (reference / benchmarking)
│
├── llm/
│   ├── base.py                    # BaseLLMClient abstract class + Message dataclass
│   └── openai_client.py           # OpenAI Chat Completions implementation
│
└── interaction/
    └── terminal.py                # TerminalInteractor: display + input helpers
```

---

## Module Reference

### `llm/base.py`

Defines `BaseLLMClient` (abstract) and `Message` dataclass. Swap `OpenAIClient` for any `BaseLLMClient` subclass to change LLM providers (Anthropic, local models, etc.).

```python
class BaseLLMClient(ABC):
    def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> str: ...
```

### `llm/openai_client.py`

Concrete implementation backed by `openai.OpenAI.chat.completions.create()`.

### `interaction/terminal.py`

`TerminalInteractor` handles all console I/O. Methods: `display()`, `display_section()`, `display_banner()`, `ask()`, `ask_questions()`, `confirm()`. Replace with a `WebInteractor` subclass to switch the interaction channel without touching agents.

### `agents/clarification_agent.py`

- `ClarificationAgent(llm, interactor, results_dir, max_rounds, dry_run)`
- `run(vague_intent, runtime_context=None) -> str`

Saves: `intent_original.txt`, `clarify_prompt.txt`, `questions_round_N.txt`, `answers_round_N.txt`, `clarified_intent.txt`.

### `agents/generator_agent.py`

- `GeneratorAgent(llm, kb_dir, topo_dir, num_candidates, temperatures, dry_run)`
- `run(clarified_intent, results_dir) -> list[dict[str, str]]`

Key helpers: `load_knowledge_base()`, `load_topo_configs()`, `generate_topology_doc()` (builds topology markdown from `.cfg` files at runtime).

### `agents/selection_agent.py`

- `SelectionAgent(llm, interactor, batfish_script_dir, max_rounds, dry_run)`
- `run(candidates, clarified_intent, results_dir, prior_clarification_qa) -> dict | None`

Returns the winning candidate dict, or `None` to trigger a recovery loop in `main.py`.

Key helpers: `_ec_prune()` (union-find), `_run_all_diffs()` (subprocess Batfish), `_find_best_pair()` (priority scan), `_generate_question()` (LLM), `_prune()` (answer-based pruning), `_do_recovery()` (writes `RuntimeContext`).

Module-level functions: `_parse_reachability()`, `_parse_advanced()`, `_is_rejection()`, `_classify_answer()`.

### `batfish/diff_analysis.py`

Runs `differentialReachability()` between two Batfish snapshots. Prints `[DIFFERENCE DETECTED]` with a counter-example packet, or `[NO DIFFERENCE FOUND]`.

```bash
python batfish/diff_analysis.py --folder results/.../candidates --c1 candidate_1 --c2 candidate_2
```

### `batfish/diff_advanced.py`

Runs `reachability()` on both snapshots and compares node sets and path counts. Prints `[DIVERGENCE DISCOVERED]` with waypointing/load-balancing details, or `All successful flows behave identically`.

```bash
python batfish/diff_advanced.py --folder results/.../candidates --c1 candidate_1 --c2 candidate_2
```

### `batfish/run_all_diffs.py`

Standalone batch runner for offline analysis. Iterates over a set of experiment folders and candidate pairs, calling both diff scripts and writing output to `comparison_results.txt` in each folder.

---

## Dry-Run Mode

`--dry-run` skips all LLM and Batfish calls and returns canned responses at each phase. Useful for testing the pipeline flow and file output structure without an API key or a running Batfish instance.

```bash
python main.py --intent "test intent" --dry-run
```

The dry-run produces real output directories with placeholder content so the full directory structure can be inspected.

---

## Extending NetSocratic

**Swap the LLM provider**

Implement `BaseLLMClient` from `llm/base.py`:

```python
class AnthropicClient(BaseLLMClient):
    def complete(self, messages, temperature=0.7, max_tokens=2048) -> str:
        ...
```

Pass it to all three agents in `main.py`.

**Change the interaction channel**

Subclass `TerminalInteractor` and override `ask()` / `ask_questions()` / `display()`. The agents accept any object with the same interface.

**Add knowledge-base files**

Drop `.md` or `.txt` files into `agents/knowledge-base/`. They are loaded automatically and injected into the Generator Agent's prompt under `## Knowledge Base`. The file `topology.md` is reserved (ignored if present — topology is always generated from `TOPO_DIR`).

**Add more Batfish analysis scripts**

Extend `SelectionAgent._run_all_diffs()` with additional script calls. Parse the new output format in `_parse_advanced()` or a new parser, and surface differences via `_find_best_pair()`.
