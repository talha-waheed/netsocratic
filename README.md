# NetSocratic

NetSocratic is a multi-agent LLM pipeline that translates vague network operator intent into precise, verified OSPF router configurations. It uses a Socratic clarification loop to resolve ambiguity, generates candidate configurations from structured routing rules, and uses [Batfish](https://www.batfish.org/) to formally verify and select among them.

---

## Architecture

The pipeline runs three agents in sequence, with an automatic recovery loop when selection fails.

```
Operator intent (vague)
        │
        ▼
┌─────────────────────┐
│  Clarification Agent │  ← targeted Q&A to resolve reachability, waypoints, ECMP counts
└─────────────────────┘
        │ clarified intent
        ▼
┌─────────────────────────────────────────┐
│            Generator Agent              │
│                                         │
│  1. Intent → Rules JSON                 │  ← N times, varied temperatures + strategies
│     (reachability / waypoint / ECMP)    │
│                                         │
│  2. Rules JSON → OSPF configs           │  ← same temperature as rules for each candidate
└─────────────────────────────────────────┘
        │ N candidate dicts (rules + configs)
        ▼
┌─────────────────────────────────────────┐
│           Selection Agent               │
│                                         │
│  1. Batfish EC-pruning                  │  ← collapse behaviourally identical candidates
│  2. Follow-up Q&A over Batfish diffs    │  ← LLM-generated questions, operator answers
│  3. Further clarified intent            │  ← selection Q&A merged back into the intent
│  4. Corrected generation                │  ← new config from refined intent
└─────────────────────────────────────────┘
        │ corrected config + further_clarified_intent (or recovery)
        ▼
┌─────────────────────────────────────────┐
│     Verification (experiments only)     │
│                                         │
│  Re-generate N candidates from the      │  ← same procedure as Phase 2
│  further-clarified intent, then count   │
│  Batfish ECs.  Goal: EC count = 1.      │
└─────────────────────────────────────────┘
        │
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
| `OPENAI_MIN_SECONDS_BETWEEN_REQUESTS` | `0` | Optional client-side delay between OpenAI calls; increase to reduce 429 rate-limit retries during experiments |
| `MAX_CLARIFY_ROUNDS` | `5` | Maximum clarification rounds before best-effort synthesis |
| `NUM_CANDIDATES` | `3` | Number of candidate rule sets the Generator produces |
| `KB_DIR` | `agents/knowledge-base` | Directory for knowledge-base Markdown/text files |
| `TOPO_DIR` | `topo` | Directory of base `.cfg` files — one per router |
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
--num-candidates N      Candidate rule sets to generate (default: 3)
--kb-dir PATH           Knowledge-base directory (default: agents/knowledge-base)
--topo-dir PATH         Base topology directory with one .cfg per router
--results-dir PATH      Root output directory (default: results)
--batfish-script-dir P  Directory containing Batfish diff scripts (default: batfish)
--max-recovery-rounds N Max full-pipeline restarts on failure (default: 2)

--no-strategies         Disable per-candidate strategy hints; rely on temperature variation only
--skip-generator        Run clarification only; stop before generation
--skip-selector         Run clarification + generation only; stop before selection
--dry-run               Skip all LLM and Batfish calls; use canned responses for testing
```

### Examples

```bash
# Full pipeline (requires Batfish running)
python main.py --intent "Athens must reach 100.0.29.0/24 via London, 3 ECMP paths"

# Clarification only
python main.py --intent "..." --skip-generator

# Clarification + generation, no Batfish verification
python main.py --intent "..." --skip-selector

# Test the full pipeline without an API key or Batfish
python main.py --intent "..." --dry-run

# Use temperature-only diversity (no strategy hints)
python main.py --intent "..." --no-strategies

# Use a different model with more candidates
python main.py --intent "..." --model gpt-4o-mini --num-candidates 5

# Load intent from a file
python main.py --intent-file intents/my_intent.txt
```

---

## Pipeline Details

### Phase 1 — Clarification Agent

The Clarification Agent removes ambiguity from the operator's intent through a structured Q&A loop. It uses two LLM passes per round:

1. **Question generation** (`CLARIFY_SYSTEM`) — the model inventories what is already known versus unknown across all three required dimensions, combines gaps into compound questions, and outputs at most 3 questions per round.
2. **Sufficiency check** (`SUFFICIENCY_SYSTEM`) — evaluates whether the accumulated Q&A is now sufficient to act on. Returns either `CLARIFIED <intent>` or `MORE_QUESTIONS <list>`.

The three dimensions that must be fully resolved before generation:

| Dimension | What is needed |
|---|---|
| **Reachability** | Exact source router name and destination subnet in CIDR notation |
| **Waypointing** | Exact waypoint router name and whether traversal is mandatory or preferred |
| **Load balancing** | Exact integer number of equal-cost (ECMP) paths |

The loop runs until the LLM declares the intent fully specified, or `--max-rounds` is reached. At max rounds the agent forces a best-effort synthesis from whatever Q&A was collected; if the LLM still returns `MORE_QUESTIONS` during synthesis, the original vague intent is used as a fallback rather than passing an unresolved question list downstream.

If a `RuntimeContext` from a previous failed attempt is provided, it is injected into every LLM prompt so the model knows what was already tried.

### Phase 2 — Generator Agent

The Generator Agent is a two-step process. **Structured rules are the primary unit of analysis** — candidates differ at the rules level, not the config level.

#### Step 1 — Intent → Rules JSON (varied per candidate)

For each of the N candidates the agent calls the LLM once to extract a structured `rules.json` from the clarified intent:

```json
{
  "reachability": { "<source_router>": ["<prefix>", ...] },
  "waypoint":     { "(<source_router>,<prefix>)": ["<waypoint_router>", ...] },
  "loadbalancing":{ "(<source_router>,<prefix>)": <integer path count> }
}
```

Two mechanisms drive diversity across candidates:

- **Candidate strategies** (default, primary) — each candidate slot is assigned one of three interpretation stances, injected as a hint into the rules-extraction prompt:

  | Candidate | Strategy | Rules behavior | Config behavior |
  |---|---|---|---|
  | 1 | Conservative | Only explicit requirements; ambiguous waypoints omitted (treated as preferred) | No cost-1000 blocking; minimal router changes |
  | 2 | Standard | Literal reading; waypoints treated as mandatory by default | Binary 1/1000 cost scale; source-router-first changes |
  | 3 | Exploratory | Infer implied requirements; waypoints always mandatory | Graduated 1/10/1000 cost scale; changes spread across intermediate routers |

- **Temperature variation** (secondary) — rules extraction temperatures default to `[0.2, 0.7, 1.0]` to add stochastic variation within each strategy's interpretation space.

Pass `--no-strategies` (or `use_strategies=False`) to disable strategy hints and rely on temperature variation alone.

#### Step 2 — Rules JSON → OSPF configs (per candidate)

Config generation runs at the **same temperature used for that candidate's rules extraction**. This means candidate 1 uses temperature 0.2 for both steps, candidate 2 uses 0.7, and candidate 3 uses 1.0. The LLM outputs only the routers whose `ip ospf cost` values must change; all other routers are copied verbatim from the base topology in `TOPO_DIR`.

Key constraints enforced in the prompt:
- Only `ip ospf cost` values may change from the base topology.
- Cost `1` = preferred path / ECMP participant.
- Cost `1000` = effectively blocked (mandatory waypointing enforcement).
- ECMP is achieved by ensuring exactly N routes share the minimum total metric.

Output per candidate:

```
results/<timestamp>/candidates/candidate_N/
├── rules.json           ← structured routing rules that drove this candidate
├── decision_summary.txt ← LLM reasoning: path enumeration, cost calculations, verification
└── configs/
    ├── Athens.cfg
    ├── London.cfg
    └── ...              (all routers — LLM-modified + base copies)
```

### Phase 3 — Selection Agent

The Selection Agent uses Batfish differences between N candidates to ask follow-up questions, then generates a fresh Network Specification from the refined intent.

#### EC-Based Pruning (Batfish)

All pairwise candidate comparisons are run via subprocess calls to the Batfish scripts:

- `batfish/diff_analysis.py` — **reachability** differences via `differentialReachability()`
- `batfish/diff_advanced.py` — **waypointing + load-balancing** differences via `reachability()` trace comparison

A union-find algorithm groups behaviorally identical candidates into equivalence classes. One representative per class survives.

#### Follow-up Q&A Loop

For each surviving pair, the LLM converts the Batfish-detected behavioral difference into a single plain-English question for the operator. The operator's answers are accumulated as new intent constraints instead of pruning to one existing candidate, so the final output can combine dimensions that were correct in different candidates.

Priority order for which difference to surface: **reachability > waypointing > load-balancing**.

After a pair has been asked about, it is skipped for subsequent rounds and the agent moves to the next most-salient distinguishable pair, preventing the same question from repeating indefinitely.

#### Further Clarified Intent + Corrected Generation

After follow-up Q&A, the selection Q&A is fed back to the LLM alongside the original clarified intent to produce a **further clarified intent** — a refined, more precise version of the intent that incorporates the operator's answers. The Generator Agent then runs once at temperature `0.0` against that further clarified intent to produce the final corrected configuration.

This further clarified intent is:
- Saved to `selection/further_clarified_intent.txt`.
- Displayed in the terminal when it differs from the original clarified intent.
- Used as the basis for the **verification phase** in experiments (see below).
- Evaluated separately in the experiment runner (`eval_further_clarified`).

The corrected configuration is saved to `selection/winner/`, including `rules.json`, `decision_summary.txt`, and full router configs. If no distinguishing Q&A occurred (e.g. only one EC survived pruning), the further clarified intent is identical to the original and the surviving candidate is saved directly.

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
├── rules.json                   # top-level rules extracted from clarified intent
├── clarify_runtime_context.txt  # (if recovery) injected context from prior run
│
├── candidates/
│   ├── candidate_1/
│   │   ├── rules.json           # rules for this specific candidate (Conservative)
│   │   ├── decision_summary.txt
│   │   └── configs/
│   │       ├── Athens.cfg
│   │       ├── London.cfg
│   │       └── ...
│   ├── candidate_2/             # Standard interpretation
│   │   └── ...
│   └── candidate_3/             # Exploratory interpretation
│       └── ...
│
├── selection/
│   ├── batfish/
│   │   ├── candidate_1_vs_candidate_2_reachability.txt
│   │   ├── candidate_1_vs_candidate_2_advanced.txt
│   │   └── ...
│   ├── selection_log.txt              # EC grouping, Q&A rounds, synthesis decisions
│   ├── further_clarified_intent.txt   # intent refined with selection Q&A preferences
│   ├── runtime_context.txt            # (if recovery) packed state for next Clarification pass
│   └── winner/
│       ├── rules.json
│       ├── decision_summary.txt
│       └── configs/
│           ├── Athens.cfg
│           ├── London.cfg
│           └── ...
│
└── verification/                      # (experiments only) re-generation from further intent
    ├── candidates/
    │   ├── candidate_1/rules.json
    │   ├── candidate_2/rules.json
    │   └── candidate_3/rules.json
    ├── batfish/                        # pairwise Batfish diffs for verification candidates
    │   └── ...
    └── selection_log.txt               # EC count (goal: 1 = fully disambiguated)
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
│   ├── generator_agent.py         # Phase 2: rules extraction + OSPF config generation
│   ├── selection_agent.py         # Phase 3: Batfish EC-pruning + distinguishing Q&A
│   └── knowledge-base/
│       ├── ground_truth_rules.md  # Canonical rules JSON format (reachability/waypoint/LB)
│       └── routing_policy.md      # OSPF cost policy guide (injected into generator prompts)
│
├── topo/
│   └── <City>.cfg                 # Base Cisco IOS router configs (one per router)
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
├── interaction/
│   ├── terminal.py                # TerminalInteractor: display + input helpers
│   └── llm_operator.py            # LLMOperator: automated operator backed by an LLM
│
└── experiments/
    └── runner.py                  # Batch experiment runner over a CSV dataset
```

---

## Module Reference

### `llm/base.py`

Defines `BaseLLMClient` (abstract) and `Message` dataclass. Swap `OpenAIClient` for any `BaseLLMClient` subclass to change LLM providers (Anthropic, local models, etc.).

```python
class BaseLLMClient(ABC):
    def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> str: ...
```

### `agents/clarification_agent.py`

- `ClarificationAgent(llm, interactor, results_dir, max_rounds, dry_run)`
- `run(vague_intent, runtime_context=None) -> str`

Saves: `intent_original.txt`, `clarify_prompt.txt`, `questions_round_N.txt`, `answers_round_N.txt`, `clarified_intent.txt`.

### `agents/generator_agent.py`

- `GeneratorAgent(llm, kb_dir, topo_dir, num_candidates, rules_temperatures, use_strategies, dry_run)`
- `run(clarified_intent, results_dir) -> list[dict[str, str]]`

Each returned dict contains `"__rules__"` (rules JSON string), `"decision_summary.txt"`, and one key per router (full config text). Both rules extraction and OSPF config generation for a given candidate use the same temperature (no separate `config_temperature` — the rules temperature drives both steps).

Module-level constants: `CANDIDATE_STRATEGIES` (list of 3 strategy dicts with `rules_hint` and `config_hint` keys — extend to add more candidates).

Key helpers: `load_knowledge_base()`, `load_topo_configs()`, `generate_topology_doc()` (builds topology markdown from `.cfg` files at runtime — no static `topology.md`).

### `agents/selection_agent.py`

- `SelectionAgent(llm, interactor, batfish_script_dir, max_rounds, dry_run)`
- `run(candidates, clarified_intent, results_dir, prior_clarification_qa) -> tuple[dict, str] | tuple[None, None]`

Returns `(winner_dict, further_clarified_intent)` on success, or `(None, None)` to trigger a recovery loop. `further_clarified_intent` incorporates selection Q&A preferences into the clarified intent (saved to `selection/further_clarified_intent.txt`).

- `count_ecs(n, cands_dir, results_dir) -> int`

Runs Batfish pairwise diffs on `n` pre-saved candidates in `cands_dir` and returns the number of behaviourally distinct equivalence classes. Used by the experiment runner's verification phase — a return value of 1 means all candidates are identical.

Key helpers: `_synthesise_further_clarified()` (LLM merge of clarified intent + selection Q&A), `_synthesise_winner()` (fresh generation from the refined intent), `_ec_prune()` (union-find), `_run_all_diffs()` (subprocess Batfish), `_find_best_pair()` (priority scan), `_generate_question()` (LLM), `_do_recovery()` (writes `RuntimeContext`).

### `interaction/llm_operator.py`

- `LLMOperator(llm, correct_spec, temperature=0.0, verbose=True)`
- Drop-in replacement for `TerminalInteractor` — implements the same interface (`display*`, `ask`, `ask_questions`, `confirm`) so it can be passed to any agent without modification.
- `ask_questions()` answers clarification questions via `_CLARIFICATION_SYSTEM`, instructing the LLM to respond as a human operator using plain English (no raw JSON exposed).
- `ask()` answers selection questions via `_SELECTION_SYSTEM`, framed as choosing between two routing behaviors.
- `verbose=True` (default) prints every Q&A exchange to stdout; set `verbose=False` to suppress output in batch runs.

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

### `experiments/runner.py`

- CLI: `python experiments/runner.py --csv <path> [options]`
- Reads `Ambiguous High Level Intent` and `Correct Formal Specification` columns from a CSV file.
- Constructs an `LLMOperator` per row using the correct spec and runs the full pipeline without a human in the loop.
- Evaluates generated rules against the correct spec at four checkpoints: post-clarification, best-of-N candidates, winner, and further-clarified intent.
- **Phase 4 — Verification**: after selection, re-runs `GeneratorAgent` using the further-clarified intent and calls `SelectionAgent.count_ecs()` to measure how many behaviourally distinct candidates result. A count of 1 means the intent was fully disambiguated.
- Writes `summary.csv` and `summary.json` incrementally after every row — partial runs are fully usable if interrupted.

Key functions: `evaluate()` (per-dimension rule comparison), `extract_rules_neutral()` (rules at temp=0.0, no strategy — post-clarification and further-clarified baselines), `_count_clarify_questions()` (parses saved question files), `_parse_selection_log()` (extracts EC count and selection round count from `selection_log.txt`), `_print_aggregate()` (terminal summary table with verification EC stats).

---

## Dry-Run Mode

`--dry-run` skips all LLM and Batfish calls and returns canned responses at each phase. Useful for testing the pipeline flow and file output structure without an API key or a running Batfish instance.

```bash
python main.py --intent "test intent" --dry-run
```

The dry-run produces real output directories with placeholder content so the full directory structure can be inspected.

---

## Experiments

The experiment runner evaluates the full pipeline against a labelled dataset, replacing the human operator with an LLM that answers questions based on a known correct specification.

### CSV format

The input CSV must have at least these two columns:

| Column | Content |
|---|---|
| `Ambiguous High Level Intent` | The vague intent string fed to the Clarification Agent |
| `Correct Formal Specification` | Ground-truth rules JSON (`reachability` / `waypoint` / `loadbalancing`) |

### Options

```
--csv PATH              Input CSV file (required)
--limit N               Stop after N rows (quick tests)
--row-ids IDS           Comma-separated row_id values to run (e.g. 1425,1426)

--model MODEL           OpenAI model (default: gpt-4o)
--max-rounds N          Max clarification rounds (default: 5)
--num-candidates N      Candidates per run (default: 3)
--kb-dir PATH           Knowledge-base directory
--topo-dir PATH         Base topology directory
--results-dir PATH      Root output directory (default: results/experiments)
--batfish-script-dir P  Batfish diff script directory
--max-recovery-rounds N Max pipeline restarts on failure (default: 2)

--no-strategies         Temperature-only diversity (no strategy hints)
--skip-selector         Skip Batfish selection; evaluate generation candidates only
--verbose-operator      Print every LLM operator Q&A exchange to stdout
--dry-run               Canned responses; no API key or Batfish needed
```

### Examples

```bash
# Run all 240 rows, skipping Batfish (fastest)
python experiments/runner.py --csv complex_example_fuzzed_dataset.csv --skip-selector

# Quick sanity check: 3 rows, dry-run
python experiments/runner.py --csv complex_example_fuzzed_dataset.csv --limit 3 --dry-run

# Run specific rows with full pipeline (requires Batfish)
python experiments/runner.py --csv complex_example_fuzzed_dataset.csv --row-ids 1425,1426

# See all operator Q&A printed to the terminal
python experiments/runner.py --csv ... --limit 5 --skip-selector --verbose-operator
```

### Output structure

```
results/experiments/<run_timestamp>/
├── summary.csv                    # one row per experiment (see metrics table below)
├── summary.json                   # same data in JSON with full diagnostics
└── <row_id>/
    └── <pipeline_timestamp>/      # same layout as a normal pipeline run
        ├── clarified_intent.txt
        ├── candidates/
        │   ├── candidate_1/rules.json
        │   ├── candidate_2/rules.json
        │   └── candidate_3/rules.json
        ├── selection/
        │   ├── winner/rules.json
        │   └── further_clarified_intent.txt
        └── verification/          # re-generation from further_clarified_intent
            ├── candidates/
            │   ├── candidate_1/rules.json
            │   ├── candidate_2/rules.json
            │   └── candidate_3/rules.json
            └── selection_log.txt  # EC count for the 3 regenerated configs
```

### Metrics

The terminal summary and `summary.csv` report accuracy at four checkpoints:

| Checkpoint | How it is produced | What it measures |
|---|---|---|
| **Post-clarify** | Rules extracted at temp=0.0, no strategy, directly from the clarified intent | Quality of the clarification phase alone |
| **Best-of-N** | `OR` across all N candidates | Upper bound on generation quality |
| **Winner** | Corrected configuration generated from the further-clarified intent, or the single surviving EC candidate when no follow-up was needed | End-to-end pipeline accuracy |
| **Furt.Clar.** | Rules extracted at temp=0.0 from the further-clarified intent | Whether selection Q&A improved the intent |

Each checkpoint reports four match flags against the ground-truth spec:

| Flag | Meaning |
|---|---|
| `exact` | All three sections (reachability, waypoint, loadbalancing) match exactly |
| `reach` | Reachability pairs match |
| `wp` | Waypoint assignments match |
| `lb` | Load-balancing path counts match |

Additional columns:

| Column | Meaning |
|---|---|
| `n_clarify_rounds` | Number of clarification rounds |
| `n_clarify_questions` | Total clarification questions asked |
| `n_ecs` | Equivalence classes after Batfish pruning (selection phase) |
| `n_selection_rounds` | Follow-up Q&A rounds in selection |
| `n_verification_ecs` | ECs among the 3 configs regenerated from the further-clarified intent; **1 = fully disambiguated** |
| `time_clarify_s` | Clarification phase wall time |
| `time_generate_s` | Generation phase wall time |
| `time_select_s` | Selection phase wall time |
| `time_verify_s` | Verification phase wall time |

The aggregate terminal table shows `Avg verification ECs` alongside a `(perfect=1: X/N)` counter — the fraction of runs where re-generation produced only one distinct behaviour.

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

**Add or modify candidate strategies**

Edit `CANDIDATE_STRATEGIES` in `agents/generator_agent.py`. Each entry is a dict with `rules_hint` (appended to the rules-extraction user message) and `config_hint` (appended to the config-generation user message). Adding a fourth entry automatically applies to candidate 4 when `--num-candidates 4` is used; candidates beyond the list length fall back to temperature-only diversity.

**Add knowledge-base files**

Drop `.md` or `.txt` files into `agents/knowledge-base/`. They are loaded automatically and injected into the Generator Agent's prompt under `## Knowledge Base`. The filename `topology.md` is reserved (ignored — topology is always generated live from `TOPO_DIR`).

**Add more Batfish analysis scripts**

Extend `SelectionAgent._run_all_diffs()` with additional script calls. Parse the new output format in `_parse_advanced()` or a new parser, and surface differences via `_find_best_pair()`.
