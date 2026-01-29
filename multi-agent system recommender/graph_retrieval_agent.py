"""
Graph Retrieval using LLM Agent
Select top-k most relevant candidate trees from candidates using LLM reasoning
"""

import json
import os
import argparse
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GraphRetrievalAgent:

    def __init__(self,
                 node_candidates_path: str,
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 top_k: int = 10,
                 max_workers: int = 5):
        """
        Initialize LLM-based graph retrieval agent

        Args:
            node_candidates_path: Path to node_candidates.json
            model_name: OpenAI model name (default: gpt-4o)
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            top_k: Number of top candidates to retrieve (default: 10)
            max_workers: Number of concurrent threads for parallel processing (default: 5)
        """

        print(f"[1/3] Initializing OpenAI client...")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name or 'gpt-4o'

        print(f"Using model: {self.model_name}")

        # Load node candidates
        print(f"\n[2/3] Loading node candidates...")
        self.node_candidates_data = self._load_json(node_candidates_path)
        total_traces = len(self.node_candidates_data.get('traces', []))
        print(f"Loaded {total_traces} traces")

        self.top_k = top_k
        self.max_workers = max_workers

        print(f"\n[3/3] Configuration:")
        print(f"  Model:            {self.model_name}")
        print(f"  Top-k candidates: {top_k}")
        print(f"  Max workers:      {max_workers} (concurrent threads)")
        print(f"\n{'='*80}\n")

    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: dict, filepath: str):
        """Save JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {filepath}")

    def _tree_to_description(self, tree: Dict, candidate_id: str, candidate_type: str) -> str:
        """
        Convert a tree structure to a detailed text description for LLM

        Args:
            tree: Tree dictionary with 'nodes' and 'edges'
            candidate_id: Candidate identifier
            candidate_type: Type of candidate ('ground_truth' or 'random')

        Returns:
            Text description of the tree
        """
        nodes = tree.get('nodes', [])
        edges = tree.get('edges', [])

        if not nodes:
            return f"**Candidate {candidate_id}** (Empty tree)"

        # Build description
        lines = [f"**Candidate {candidate_id}**"]
        lines.append(f"- Type: {candidate_type}")
        lines.append(f"- Size: {len(nodes)} nodes, {len(edges)} edges")

        # Root node
        root = nodes[0]
        lines.append(f"- Root Task: {root.get('task', 'N/A')}")

        # Build adjacency for tree structure
        adjacency = {}
        for source, target in edges:
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)

        # Node details
        if len(nodes) > 1:
            lines.append("- Child Nodes:")
            for i, node in enumerate(nodes[1:], 1):
                node_id = node.get('node_id', '')
                task = node.get('task', 'N/A')
                lines.append(f"  {i}. [{node_id}] {task}")

        # Edge structure
        if edges:
            lines.append("- Call Graph Structure:")
            for source, target in edges[:5]:  # Limit to first 5 edges
                lines.append(f"  {source} → {target}")
            if len(edges) > 5:
                lines.append(f"  ... ({len(edges) - 5} more edges)")

        return "\n".join(lines)

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call LLM with the given prompt

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            LLM response text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return ""

    def _select_top_candidates_with_llm(self,
                                        node_task: str,
                                        candidates: List[Dict],
                                        top_k: int = 10) -> Tuple[List[Dict], bool]:
        """
        Select top-k candidates using LLM reasoning

        Args:
            node_task: The task description of the current node
            candidates: List of candidate dictionaries (ground_truth + random_candidates)
            top_k: Number of top candidates to select

        Returns:
            Tuple of (selected_candidates, ground_truth_in_top_k)
        """
        # Prepare candidate descriptions
        candidate_descriptions = []
        candidate_mapping = {}  # Map candidate_id to full candidate info

        for i, candidate in enumerate(candidates):
            cand_id = candidate.get('candidate_id', f'cand_{i}')
            cand_type = candidate.get('type', 'unknown')
            tree = candidate.get('tree', {})

            description = self._tree_to_description(tree, i+1, cand_type)
            candidate_descriptions.append(description)
            candidate_mapping[i+1] = candidate  # 1-indexed

        # Build prompt for LLM
        system_prompt = """You are an expert AI system architect specializing in multi-agent systems.
Your task is to analyze a given task and select the most relevant agent system architectures (represented as calling graphs) that would best accomplish that task.

Consider these factors when selecting:
1. **Task-Graph Alignment**: Does the graph structure match the task requirements?
2. **Completeness**: Does the graph have all necessary components?
3. **Efficiency**: Is the graph structure efficient (not overly complex)?
4. **Logical Flow**: Do the node connections make sense for the task?

Output ONLY the candidate numbers (1-indexed) that you select, separated by commas.
Example output: 1, 3, 7, 12, 15"""

        candidates_text = "\n\n".join(candidate_descriptions)

        user_prompt = f"""**Task**: {node_task}

**Available Candidates** ({len(candidates)} total):

{candidates_text}

Please select the top {top_k} most relevant candidates for this task.
Output format: Just the numbers separated by commas (e.g., "1, 5, 8, 12, 14")"""

        # Call LLM
        response = self._call_llm(user_prompt, system_prompt)

        # Parse response to get selected candidate numbers
        selected_numbers = []
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            selected_numbers = [int(n) for n in numbers if 1 <= int(n) <= len(candidates)][:top_k]
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"  Response was: {response[:200]}")
            # Fallback: select first top_k
            selected_numbers = list(range(1, min(top_k + 1, len(candidates) + 1)))

        # If we didn't get enough selections, fill with remaining candidates
        if len(selected_numbers) < top_k:
            remaining = [i for i in range(1, len(candidates) + 1) if i not in selected_numbers]
            selected_numbers.extend(remaining[:top_k - len(selected_numbers)])

        # Build result
        selected_candidates = []
        ground_truth_in_top_k = False

        for num in selected_numbers:
            if num in candidate_mapping:
                candidate = candidate_mapping[num]
                tree = candidate.get('tree', {})

                selected_candidates.append({
                    'candidate_id': candidate.get('candidate_id', ''),
                    'type': candidate.get('type', ''),
                    'similarity': 1.0 - (selected_numbers.index(num) * 0.05),  # Simulate ranking score
                    'tree_size': len(tree.get('nodes', [])),
                    'llm_rank': selected_numbers.index(num) + 1
                })

                if candidate.get('type') == 'ground_truth':
                    ground_truth_in_top_k = True

        return selected_candidates, ground_truth_in_top_k

    def _process_single_node(self, node_cand: Dict) -> Tuple[Dict, bool, bool]:
        """
        Process a single node (for parallel execution)

        Args:
            node_cand: Node candidate data

        Returns:
            Tuple of (node_selection, gt_in_top_k, ground_truth_added_flag)
        """
        node_id = node_cand.get('node_id', '')
        node_task = node_cand.get('node_task', '')

        # Combine ground truth and random candidates
        all_candidates = [node_cand['ground_truth']]
        all_candidates.extend(node_cand.get('random_candidates', []))

        # Select top-k candidates using LLM
        top_candidates, gt_in_top_k = self._select_top_candidates_with_llm(
            node_task, all_candidates, top_k=self.top_k
        )

        # Prepare final selection
        final_selection = top_candidates.copy()

        # Add ground truth if not in top-k
        ground_truth_added_flag = False
        if not gt_in_top_k:
            gt_candidate = node_cand['ground_truth']
            gt_tree = gt_candidate.get('tree', {})

            final_selection.append({
                'candidate_id': gt_candidate.get('candidate_id', ''),
                'type': 'ground_truth',
                'similarity': 0.5,
                'tree_size': len(gt_tree.get('nodes', [])),
                'added_manually': True,
                'llm_rank': self.top_k + 1
            })
            ground_truth_added_flag = True

        node_selection = {
            'node_id': node_id,
            'node_task': node_task,
            'selected_candidates': final_selection,
            'total_selected': len(final_selection),
            'ground_truth_in_topk': gt_in_top_k,
            'ground_truth_added': ground_truth_added_flag
        }

        return node_selection, gt_in_top_k, ground_truth_added_flag

    def retrieve_for_all_nodes(self, output_path: str = None, use_parallel: bool = True) -> Dict:
        """
        Retrieve top candidates for all nodes using LLM

        Args:
            output_path: Path to save results
            use_parallel: Whether to use parallel processing (default: True)

        Returns:
            Dictionary with retrieval results
        """
        print(f"Retrieving top candidates using LLM...")
        print(f"  Parallel processing: {'Enabled' if use_parallel else 'Disabled'}")
        print(f"{'='*80}\n")

        results = {
            'metadata': {
                'method': 'llm_agent_selection',
                'llm_provider': 'openai',
                'model': self.model_name,
                'top_k': self.top_k,
                'max_workers': self.max_workers if use_parallel else 1,
                'total_traces': 0,
                'total_nodes': 0,
                'ground_truth_in_topk': 0,
                'ground_truth_added': 0
            },
            'traces': []
        }

        traces = self.node_candidates_data.get('traces', [])
        results['metadata']['total_traces'] = len(traces)

        total_nodes = 0
        gt_in_topk = 0
        gt_added = 0

        for trace_idx, trace in enumerate(traces):
            trace_id = trace.get('trace_id', '')
            node_candidates_list = trace.get('node_candidates', [])

            print(f"Processing trace {trace_idx + 1}/{len(traces)}: {trace_id}")

            trace_results = {
                'trace_id': trace_id,
                'node_selections': []
            }

            if use_parallel and len(node_candidates_list) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_node = {
                        executor.submit(self._process_single_node, node_cand): node_cand
                        for node_cand in node_candidates_list
                    }

                    # Collect results with progress bar
                    with tqdm(total=len(node_candidates_list),
                             desc=f"  Trace {trace_idx + 1}",
                             leave=False) as pbar:
                        for future in as_completed(future_to_node):
                            try:
                                node_selection, gt_in_top_k, ground_truth_added_flag = future.result()
                                trace_results['node_selections'].append(node_selection)
                                total_nodes += 1

                                if gt_in_top_k:
                                    gt_in_topk += 1
                                if ground_truth_added_flag:
                                    gt_added += 1

                            except Exception as e:
                                node_cand = future_to_node[future]
                                print(f"  ✗ Error processing node {node_cand.get('node_id', 'unknown')}: {e}")

                            pbar.update(1)
            else:
                # Sequential processing
                for node_idx, node_cand in enumerate(node_candidates_list):
                    node_id = node_cand.get('node_id', '')
                    print(f"  Node {node_idx + 1}/{len(node_candidates_list)}: {node_id}")

                    try:
                        node_selection, gt_in_top_k, ground_truth_added_flag = self._process_single_node(node_cand)
                        trace_results['node_selections'].append(node_selection)
                        total_nodes += 1

                        if gt_in_top_k:
                            gt_in_topk += 1
                        if ground_truth_added_flag:
                            gt_added += 1

                    except Exception as e:
                        print(f"  ✗ Error processing node {node_id}: {e}")

            results['traces'].append(trace_results)

        # Update metadata
        results['metadata']['total_nodes'] = total_nodes
        results['metadata']['ground_truth_in_topk'] = gt_in_topk
        results['metadata']['ground_truth_added'] = gt_added

        print(f"\n{'='*80}")
        print(f"Retrieval complete!")
        print(f"  Total traces: {len(traces)}")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Ground truth in top-{self.top_k}: {gt_in_topk} ({gt_in_topk/total_nodes*100:.1f}%)")
        print(f"  Ground truth added: {gt_added} ({gt_added/total_nodes*100:.1f}%)")
        print(f"{'='*80}\n")

        # Save to file if output path is provided
        if output_path:
            self._save_json(results, output_path)

        return results


def _find_data_file(filename: str) -> str:
    """Smart path resolution to find data files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    search_paths = [
        filename,
        os.path.join(script_dir, '..', 'output', filename),
        os.path.join(script_dir, filename),
        os.path.abspath(filename)
    ]

    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    return search_paths[0]


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Graph Retrieval using LLM Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses gpt-4o by default)
  python graph_retrieval_agent.py

  # Use different OpenAI model
  python graph_retrieval_agent.py --model gpt-4o-mini

  # Specify custom paths
  python graph_retrieval_agent.py --node_candidates /path/to/node_candidates.json \\
                                  --output_dir ./output

  # Adjust top-k and parallelism
  python graph_retrieval_agent.py --top_k 5 --max_workers 10

  # Disable parallel processing
  python graph_retrieval_agent.py --no_parallel
        """
    )

    parser.add_argument(
        '--node_candidates',
        type=str,
        default=None,
        help='Path to node_candidates.json (default: auto-detect in ../output/)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: ../output)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='OpenAI model name (default: gpt-4o)'
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='OpenAI API key (default: read from OPENAI_API_KEY environment variable)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top candidates to retrieve per node (default: 10)'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        default=10,
        help='Number of concurrent threads for parallel processing (default: 5)'
    )

    parser.add_argument(
        '--no_parallel',
        action='store_true',
        help='Disable parallel processing (use sequential mode)'
    )

    args = parser.parse_args()

    # Auto-detect node_candidates file if not specified
    if args.node_candidates is None:
        args.node_candidates = _find_data_file('node_candidates.json')

    # Set output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'graph_selection_results.json')

    # Initialize retriever agent
    agent = GraphRetrievalAgent(
        node_candidates_path=args.node_candidates,
        model_name=args.model,
        api_key=args.api_key,
        top_k=args.top_k,
        max_workers=args.max_workers
    )

    # Retrieve candidates
    results = agent.retrieve_for_all_nodes(output_path, use_parallel=not args.no_parallel)

    print(f"\ngraph retrieval complete!")
    print(f"  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
