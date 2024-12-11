# McToT: Monte Carlo Enhanced Tree of Thought

A framework to enhance the reasoning capability of LLM. Based off [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291) by Jieyi Long.


## System Architecture

Please refer to our demo video: [https://youtu.be/91Z18zIw19M](https://youtu.be/91Z18zIw19M)

## Demo
```
cd tree-of-thought-puzzle-solver
```
touch `config.yaml`
```
chatbot:
    type: "openai"
    max_context_length: 8000
    include_chat_history_in_query: false
openai:
    model: "gpt-4o"
    version: "2024-02-15-preview"
```
touch `.env`
```
LANGCHAIN_TRACING_V2=
LANGCHAIN_ENDPOINT=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=

AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
```

run
```
python run_tot.py -data "data/benchmarks/sudoku/3x3_sudoku_puzzles.json" -prompt_type "policy" | tee output_3x3_policy.log

```

* note: Due to limitied time, the demo code is currently coupled with Azure OpenAI. We will decouple it to incorate more generic LLM endpoints in the future.

## How to reproduce our work

1. Generate training data with MCTS
2. Train value model with LSTM
3. Load the model and run ToT

### How to generate training data with MCTS

From project root, run
```sh
python -m mcts.main
```
```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--iterations ITERATIONS] [--shuffle SHUFFLE]

Run MCTS on Sudoku puzzles and export results.

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to JSON file containing initial Sudoku states. If not provided, a default state will be used.
  --output OUTPUT       Path to the output directory.
  --iterations ITERATIONS
                        Number of MCTS iterations to perform per initial state.
  --shuffle SHUFFLE     Shuffle the initial states before running MCTS.
```

### How to train Value Model

Please refer to `tree-of-though-puzzle-solver/ValueModel/DL11785_ValueModel_Train_LSTM.ipynb`
