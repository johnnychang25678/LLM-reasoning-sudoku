# Tree-of-Thought Implementation
An implementation of Tree-of-Thought framework inspired by the research paper [Large Language Model Guided Tree-of-Thought](https://arxiv.org/abs/2305.08291) by Jieyi Long.

## How to run
Install required dependencies:
```
pip install -r requirements.txt
```
Add .env file, please reference `.env.example`:
```
cp .env.example .env
```
Run from root:
```
python main.py -data "puzzles/4x4_sudoku_puzzles.json" -type "rule" -puzzle_size 4 -log 4x4.log
```