# Agent System Recommender

A comprehensive framework for tool and agent system recommendation using Learning-to-Rank (LTR) models with BERT embeddings.

## Overview

This project implements two recommendation systems:
- **Single Agent Recommender**: Tool selection for single-agent tasks
- **Multi-Agent System Recommender**: Graph-based agent system selection for multi-agent tasks

Both systems use a two-stage approach:
1. **Stage 1**: Candidate retrieval using BERT embedding similarity
2. **Stage 2**: Learning-to-Rank model for final selection

## Project Structure

```
.
├── data/                          # Dataset directory
│   ├── agent-data_protocol/
│   ├── Agents_Failure_Attribution/
│   ├── GTA/
│   ├── GUI-360/
│   ├── MCPToolBenchPP/
│   ├── MedAgentBench/
│   ├── Seal-Tools/
│   └── trail-benchmark/
├── single agent recommender/      # Single agent tool selection
│   ├── tool_selection.py         # Stage 1: Embedding-based tool retrieval
│   ├── learning_to_rank.py       # Stage 2: LTR model training
│   └── visualize_results.py      # Results visualization
├── multi-agent system recommender/ # Multi-agent system selection
│   ├── generate_node_candidates.py  # Generate candidate systems
│   ├── graph_retrieval.py          # Stage 1: Graph retrieval
│   ├── learning_to_rank.py         # Stage 2: LTR model training
│   └── visualize_results.py        # Results visualization
├── output/                        # Results output directory
└── figure/                        # Generated visualizations

```

## Features

### Single Agent Recommender
- BERT-based semantic similarity for tool selection
- Linear Learning-to-Rank with 4 features:
  - φ_rel: Tool-query relevance
  - φ_hist: Historical reliability
  - φ_coop: Graph-aware compatibility
  - φ_struct: Structural utility

### Multi-Agent System Recommender
- Graph-based agent system representation
- Random tree generation for candidates
- BERT-based graph retrieval
- Linear Learning-to-Rank for system selection

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- matplotlib
- numpy
- scipy

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Single Agent Recommender

#### Step 1: Tool Selection (Embedding-based Retrieval)
```bash
cd "single agent recommender"
python tool_selection.py --tool_pool ../data/your_dataset/tool_pool.json \
                        --calling_graph ../data/your_dataset/tool_calling_graphs.json \
                        --output_dir ../output
```

#### Step 2: Learning-to-Rank Training
```bash
python learning_to_rank.py --tool_pool ../data/your_dataset/tool_pool.json \
                           --calling_graph ../data/your_dataset/tool_calling_graphs.json \
                           --output_dir ../output
```

#### Step 3: Visualize Results
```bash
python visualize_results.py --output_dir ../output
```

### Multi-Agent System Recommender

#### Step 1: Generate Node Candidates
```bash
cd "multi-agent system recommender"
python generate_node_candidates.py --tool_pool ../data/your_dataset/tool_pool.json \
                                   --calling_graph ../data/your_dataset/tool_calling_graphs.json \
                                   --n_random 10 \
                                   --output_dir ../output
```

#### Step 2: Graph Retrieval
```bash
python graph_retrieval.py --node_candidates ../output/node_candidates.json \
                         --output_dir ../output
```

#### Step 3: Learning-to-Rank Training
```bash
python learning_to_rank.py --graph_selection ../output/graph_selection_results.json \
                           --node_candidates ../output/node_candidates.json \
                           --output_dir ../output
```

#### Step 4: Visualize Results
```bash
python visualize_results.py --output_dir ../output
```

## Dataset Format

### tool_pool.json
```json
{
  "tools": {
    "tool_name": {
      "description": "Tool description",
      "inputs": "Input specification"
    }
  }
}
```

### tool_calling_graphs.json
```json
{
  "traces": [
    {
      "trace_id": "trace_1",
      "nodes": {
        "node_1": {
          "task": "Task description",
          "input_spec": "{}",
          "output_spec": "result"
        }
      },
      "edges": [["node_1", "node_2"]],
      "decisions": [
        {
          "node": "node_1",
          "candidates": ["tool_1", "tool_2"],
          "chosen": "tool_1"
        }
      ]
    }
  ]
}
```

## Model Configuration

### Learning-to-Rank Parameters
- **Learning Rate**: 0.01 (adjustable via `--lr`)
- **Epochs**: 100 (adjustable via `--epochs`)
- **L2 Regularization**: 0.001 (adjustable via `--lambda_reg`)
- **BERT Model**: sentence-transformers/all-MiniLM-L6-v2 (adjustable via `--model`)

### Alternative BERT Models
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `sentence-transformers/all-mpnet-base-v2` (higher accuracy)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

## Output Files

### Single Agent
- `tool_selection_results.json`: Stage 1 retrieval results
- `ltr_model_weights.json`: Learned feature weights
- `ltr_test_results.json`: Detailed test results

### Multi-Agent
- `node_candidates.json`: Generated candidate systems
- `graph_selection_results.json`: Stage 1 retrieval results
- `graph_ltr_model_weights.json`: Learned feature weights
- `graph_ltr_test_results.json`: Detailed test results

### Visualizations
All visualizations are saved to the `figure/` directory in PNG format.

## Evaluation Metrics

- **Top-1 Accuracy**: Percentage of queries where the correct tool/system is ranked first
- **Top-3 Accuracy**: Percentage of queries where the correct tool/system is in top 3
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct answers

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{agent_system_recommender,
  title={Agent System Recommender: Learning-to-Rank for Tool and System Selection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/agent-system-recommender}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- BERT embeddings from Hugging Face Transformers
- Sentence-BERT for semantic similarity

