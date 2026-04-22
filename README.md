# evorun

This is me messing around with automatic optimization with LLMs, based on [MLEvolve](https://github.com/InternScience/MLEvolve) but with claude and standard project structure.

## Project structure

```
my_project/
├── experiment/   # folder with code/params the LLM edits
├── eval.py       # default eval script 
├── pixi.toml     # default environment for running eval.py
├── TASK.md       # describes the task and objective
└── config.toml   # optional, see config.example.toml
```

By default it runs `pixi run python eval.py` and expects it to print something like `{"score": <float>, "description": "more detailed metrics as string"}`
in the last line to stdout. Any other output is allowed, just make sure the very last line is structured like that!
You can configure the eval command in the config.toml with the `eval_cmd` parameter.

You can have any other files in the `my_project` folder, the LLM will edit only the `experiment/` subfolder.

One nice structure that I use is having a separate `code/` subfolder for my codebase, and then symlink the relevant files from `experiment/` to `code/` so the structure is maintained for my project.

## Commands

- `evorun run` — run the optimization loop on a project directory
- `evorun viz` — start the web visualization server to inspect the search tree
- `evorun init` — scaffold a new project with starter config, TASK.md, and eval.py
- `evorun restore` — restore the codebase from a snapshot (best, root, or specific node)
- `evorun tree` — print a tree summary of the run with scores and edit summaries
- `evorun history` — print iterations in chronological order with scores and edit summaries
