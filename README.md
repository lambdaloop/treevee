# evorun

This is me messing around with automatic optimization with LLMs, based on [MLEvolve](https://github.com/InternScience/MLEvolve) but with claude and standard project structure.

## Commands

- `evorun run` — run the optimization loop on a project directory
- `evorun viz` — start the web visualization server to inspect the search tree
- `evorun init` — scaffold a new project with starter config, TASK.md, and eval.py
- `evorun restore` — restore the codebase from a snapshot (best, root, or specific node)
- `evorun tree` — print a tree summary of the run with scores and edit summaries
- `evorun history` — print iterations in chronological order with scores and edit summaries
