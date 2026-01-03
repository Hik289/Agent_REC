# Data Source

## Dataset Information

The tool calling graph data for this project is sourced from Hugging Face:

**Dataset URL**: [https://huggingface.co/datasets/xsong69/Tool_calling_graphs](https://huggingface.co/datasets/xsong69/Tool_calling_graphs)

## Dataset Structure

The dataset contains multiple subdirectories, each representing a different benchmark:

- `agent-data_protocol/`
- `Agents_Failure_Attribution/`
- `GTA/`
- `GUI-360/`
- `MCPToolBenchPP/`
- `MedAgentBench/`
- `Seal-Tools/`
- `trail-benchmark/`

## Data Format

Each subdirectory contains two JSON files:

### 1. tool_pool.json

Tool pool file containing descriptions of available tools:

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

### 2. tool_calling_graphs.json

Tool calling graph file containing task execution traces:

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

## Usage

This data is used to train and evaluate two recommendation systems:

1. **Single Agent Recommender**: Tool selection for single-agent tasks
2. **Multi-Agent System Recommender**: Graph-based agent system selection for multi-agent tasks

## License

MIT License

## Citation

If you use this dataset in your research, please cite the original source.
