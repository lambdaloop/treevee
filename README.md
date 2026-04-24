# treevee - optimize anything with a scoring function, tree search, and large language models 

This is me messing around with automatic optimization with LLMs, based on [MLEvolve](https://github.com/InternScience/MLEvolve) but with claude and standard project structure.

![Screenshot of treevee interface](./header.png)


## How it works

Given a scoring function, a task description, and code to modify, treevee will conduct a tree search over possible solutions to improve the score. At each step, it will generate a new solution by asking large language models (LLM) to propose changes and implement them.

More specifically, the flow is as follows:
1. Pick a node from the tree
2. Ask a planner LLM to provide suggestions to improve the node, debug it, or run fusion between nodes
3. Run editor LLM to implement the suggestions
4. Run the evaluation command to compute the score
5. Repeat

This is not a new idea. The treevee code leans heavily on the [MLEvolve](https://github.com/InternScience/MLEvolve) implementation in particular. Some other interesting references are [AIDE ML](https://github.com/WecoAI/aideml) or the [AI scientist paper](https://arxiv.org/abs/2509.06503) from Google Research. 

Combining LLM mutations with tree search and a concrete score function makes them more robust to failures and allows for iterative exploration of a solution space. 

## Why treevee? 

I couldn't fully wrap my mind around MLEvolve or AIDE ML, so I made my own thing. 

Some opinionated differences relative to other frameworks:
- clear project structure
- planner / editor separation
- only one solution running at a time
- properly sandboxed 

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

## Examples

## Commands

- `treevee run` — run the optimization loop on a project directory
- `treevee viz` — start the web visualization server to inspect the search tree
- `treevee init` — scaffold a new project with starter config, TASK.md, and eval.py
- `treevee restore` — restore the codebase from a snapshot (best, root, or specific node)
- `treevee tree` — print a tree summary of the run with scores and edit summaries
- `treevee history` — print iterations in chronological order with scores and edit summaries

## Safety

bubblewrap


## Questions

### 
