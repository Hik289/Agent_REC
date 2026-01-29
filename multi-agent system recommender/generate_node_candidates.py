#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Node Candidates for Multi-Agent System
For each node in the calling graph, generate N perturbed calling trees using two methods:
  1. Node perturbation: modify node tools (replace/delete/add)
  2. Topology perturbation: modify graph structure (edges only)
Plus the ground truth tree.
"""

import json
import random
import os
import argparse
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class CallTreeGenerator:
    """Generate perturbed calling trees using hybrid perturbation methods and extract ground truth trees"""
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str, n_random: int = 20):
        """
        Initialize the generator
        
        Args:
            tool_pool_path: Path to tool_pool.json
            calling_graph_path: Path to tool_calling_graphs.json
            n_random: Number of candidate trees to generate per node (half node, half topology perturbation)
        """
        print(f"\n{'='*80}")
        print("Node Candidate Generator for Multi-Agent System")
        print(f"{'='*80}\n")
        
        print(f"[1/3] Loading tool pool...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.tools = list(self.tool_pool.get('tools', {}).keys())
        print(f"  ✓ Loaded {len(self.tools)} tools")
        
        print(f"\n[2/3] Loading calling graph...")
        self.calling_graph = self._load_json(calling_graph_path)
        self.traces = self.calling_graph.get('traces', [])
        print(f"  ✓ Loaded {len(self.traces)} traces")
        
        self.n_random = n_random
        print(f"\n[3/3] Configuration:")
        print(f"  Candidate trees per node: {n_random}")
        print(f"  - Node perturbations: {n_random // 2} (modify node tools)")
        print(f"  - Topology perturbations: {n_random - n_random // 2} (modify edges)")
        print(f"  Total candidates per node: {n_random + 1} (including ground truth)")
        print(f"  Generation methods:")
        print(f"    • Node: replace/delete/add nodes with different tools")
        print(f"    • Topology: delete/add/rewire edges")
        print(f"\n{'='*80}\n")
    
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_json(self, data: dict, filepath: str):
        """Save JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to: {filepath}")
    
    def _build_adjacency_list(self, edges: List[List[str]]) -> Dict[str, List[str]]:
        """Build adjacency list from edge list"""
        adj = defaultdict(list)
        for source, target in edges:
            adj[source].append(target)
        return dict(adj)
    
    def _extract_subtree(self, node_id: str, nodes: Dict, adjacency: Dict) -> Dict:
        """
        Extract the subtree rooted at node_id (BFS traversal)
        
        Returns:
            Dictionary representing the subtree with nodes and edges
        """
        if node_id not in nodes:
            return {'nodes': [], 'edges': []}
        
        # BFS to find all descendants
        visited = set()
        queue = [node_id]
        subtree_nodes = []
        subtree_edges = []
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            subtree_nodes.append({
                'node_id': current,
                'task': nodes[current].get('task', ''),
                'input_spec': nodes[current].get('input_spec', ''),
                'output_spec': nodes[current].get('output_spec', '')
            })
            
            # Add children
            if current in adjacency:
                for child in adjacency[current]:
                    subtree_edges.append([current, child])
                    queue.append(child)
        
        return {
            'nodes': subtree_nodes,
            'edges': subtree_edges
        }
    
    def _generate_perturbed_tree(self, ground_truth_tree: Dict, perturbation_type: str = 'replace') -> Dict:
        """
        Generate a perturbed tree by modifying one node from the ground truth tree
        
        Args:
            ground_truth_tree: The ground truth tree to perturb
            perturbation_type: Type of perturbation ('replace', 'delete', 'add')
            
        Returns:
            Dictionary representing a perturbed tree with nodes and edges
        """
        import copy
        
        # Deep copy to avoid modifying original
        perturbed_tree = copy.deepcopy(ground_truth_tree)
        
        # If tree is empty, return empty tree
        if not perturbed_tree['nodes']:
            return perturbed_tree
        
        # Choose perturbation type with probabilities
        if perturbation_type == 'auto':
            perturbation_type = random.choices(
                ['replace', 'delete', 'add'],
                weights=[0.7, 0.2, 0.1]  # 70% replace, 20% delete, 10% add
            )[0]
        
        if perturbation_type == 'replace' and len(perturbed_tree['nodes']) > 0:
            # Replace a random node's tool
            node_idx = random.randint(0, len(perturbed_tree['nodes']) - 1)
            selected_node = perturbed_tree['nodes'][node_idx]
            
            # Get current tool from task
            current_task = selected_node.get('task', '')
            
            # Select a different random tool
            new_tool = random.choice(self.tools)
            selected_node['task'] = f'Call {new_tool}'
            selected_node['node_id'] = f"perturbed_{selected_node['node_id']}"
            
        elif perturbation_type == 'delete' and len(perturbed_tree['nodes']) > 1:
            # Delete a random non-root node
            # Don't delete the root (first node)
            node_idx = random.randint(1, len(perturbed_tree['nodes']) - 1)
            deleted_node_id = perturbed_tree['nodes'][node_idx]['node_id']
            
            # Remove the node
            perturbed_tree['nodes'].pop(node_idx)
            
            # Remove edges connected to this node
            perturbed_tree['edges'] = [
                edge for edge in perturbed_tree['edges']
                if edge[0] != deleted_node_id and edge[1] != deleted_node_id
            ]
            
        elif perturbation_type == 'add' and len(perturbed_tree['nodes']) > 0:
            # Add a new child to a random node
            parent_idx = random.randint(0, len(perturbed_tree['nodes']) - 1)
            parent_node = perturbed_tree['nodes'][parent_idx]
            
            # Create new node
            new_tool = random.choice(self.tools)
            new_node_id = f"added_{random.randint(1000, 9999)}"
            new_node = {
                'node_id': new_node_id,
                'task': f'Call {new_tool}',
                'input_spec': '{}',
                'output_spec': 'result from environment'
            }
            
            perturbed_tree['nodes'].append(new_node)
            perturbed_tree['edges'].append([parent_node['node_id'], new_node_id])
        
        return perturbed_tree
    
    def _generate_topology_perturbed_tree(self, ground_truth_tree: Dict, perturbation_type: str = 'auto') -> Dict:
        """
        Generate a perturbed tree by modifying topology only (edges), keeping nodes unchanged
        
        Args:
            ground_truth_tree: The ground truth tree to perturb
            perturbation_type: Type of perturbation ('delete_edge', 'add_edge', 'rewire_edge')
            
        Returns:
            Dictionary representing a topology-perturbed tree
        """
        import copy
        
        # Deep copy to avoid modifying original
        perturbed_tree = copy.deepcopy(ground_truth_tree)
        
        # If tree is empty or has only one node, return as is
        if len(perturbed_tree['nodes']) <= 1:
            return perturbed_tree
        
        # Choose perturbation type with probabilities
        if perturbation_type == 'auto':
            perturbation_type = random.choices(
                ['delete_edge', 'add_edge', 'rewire_edge'],
                weights=[0.3, 0.3, 0.4]  # 30% delete, 30% add, 40% rewire
            )[0]
        
        if perturbation_type == 'delete_edge' and len(perturbed_tree['edges']) > 0:
            # Delete a random edge
            if len(perturbed_tree['edges']) > 0:
                edge_idx = random.randint(0, len(perturbed_tree['edges']) - 1)
                removed_edge = perturbed_tree['edges'].pop(edge_idx)
        
        elif perturbation_type == 'add_edge' and len(perturbed_tree['nodes']) >= 2:
            # Add a new edge between two nodes that aren't directly connected
            node_ids = [node['node_id'] for node in perturbed_tree['nodes']]
            existing_edges_set = set((e[0], e[1]) for e in perturbed_tree['edges'])
            
            # Find potential new edges
            potential_edges = []
            for i, parent_id in enumerate(node_ids):
                for j, child_id in enumerate(node_ids):
                    if i != j and (parent_id, child_id) not in existing_edges_set:
                        # Avoid creating cycles (simple check)
                        potential_edges.append([parent_id, child_id])
            
            if potential_edges:
                new_edge = random.choice(potential_edges)
                perturbed_tree['edges'].append(new_edge)
        
        elif perturbation_type == 'rewire_edge' and len(perturbed_tree['edges']) > 0:
            # Rewire an existing edge: change either source or target
            edge_idx = random.randint(0, len(perturbed_tree['edges']) - 1)
            old_edge = perturbed_tree['edges'][edge_idx]
            
            node_ids = [node['node_id'] for node in perturbed_tree['nodes']]
            
            # Randomly choose to change source or target
            if random.random() < 0.5 and len(node_ids) > 1:
                # Change source (parent)
                new_source = random.choice([nid for nid in node_ids if nid != old_edge[1]])
                perturbed_tree['edges'][edge_idx] = [new_source, old_edge[1]]
            else:
                # Change target (child)
                new_target = random.choice([nid for nid in node_ids if nid != old_edge[0]])
                perturbed_tree['edges'][edge_idx] = [old_edge[0], new_target]
        
        return perturbed_tree
    
    def generate_candidates_for_all_nodes(self, output_path: str = None) -> Dict:
        """
        Generate candidates for all nodes in all traces
        
        Returns:
            Dictionary with node candidates
        """
        print(f"Generating node candidates...")
        print(f"{'='*80}\n")
        
        all_candidates = {
            'metadata': {
                'n_random_trees': self.n_random,
                'n_node_perturbations': self.n_random // 2,
                'n_topology_perturbations': self.n_random - self.n_random // 2,
                'total_candidates_per_node': self.n_random + 1,
                'total_traces': len(self.traces),
                'generation_method': 'hybrid_perturbation (node + topology)'
            },
            'traces': []
        }
        
        total_nodes_processed = 0
        
        for trace_idx, trace in enumerate(self.traces):
            trace_id = trace.get('trace_id', f'trace_{trace_idx}')
            nodes = trace.get('nodes', {})
            edges = trace.get('edges', [])
            
            if (trace_idx + 1) % 10 == 0 or trace_idx == 0:
                print(f"Processing trace {trace_idx + 1}/{len(self.traces)}: {trace_id}")
            
            # Build adjacency list for this trace
            adjacency = self._build_adjacency_list(edges)
            
            # Generate candidates for each node
            trace_candidates = {
                'trace_id': trace_id,
                'node_candidates': []
            }
            
            for node_id in nodes.keys():
                # Extract ground truth subtree
                ground_truth_tree = self._extract_subtree(node_id, nodes, adjacency)
                
                # Generate perturbed trees: half node perturbation, half topology perturbation
                random_trees = []
                n_node_perturb = self.n_random // 2  # 10 node perturbations
                n_topo_perturb = self.n_random - n_node_perturb  # 10 topology perturbations
                
                # Generate node perturbations (modify node tools)
                for i in range(n_node_perturb):
                    perturbed_tree = self._generate_perturbed_tree(
                        ground_truth_tree,
                        perturbation_type='auto'
                    )
                    random_trees.append({
                        'candidate_id': f'{node_id}_random_{i}',
                        'type': 'random',
                        'perturbation_method': 'node',
                        'tree': perturbed_tree
                    })
                
                # Generate topology perturbations (modify edges only)
                for i in range(n_topo_perturb):
                    topo_perturbed_tree = self._generate_topology_perturbed_tree(
                        ground_truth_tree,
                        perturbation_type='auto'
                    )
                    random_trees.append({
                        'candidate_id': f'{node_id}_random_{n_node_perturb + i}',
                        'type': 'random',
                        'perturbation_method': 'topology',
                        'tree': topo_perturbed_tree
                    })
                
                # Combine all candidates
                node_candidates = {
                    'node_id': node_id,
                    'node_task': nodes[node_id].get('task', ''),
                    'ground_truth': {
                        'candidate_id': f'{node_id}_ground_truth',
                        'type': 'ground_truth',
                        'tree': ground_truth_tree
                    },
                    'random_candidates': random_trees,
                    'total_candidates': len(random_trees) + 1
                }
                
                trace_candidates['node_candidates'].append(node_candidates)
                total_nodes_processed += 1
            
            all_candidates['traces'].append(trace_candidates)
        
        all_candidates['metadata']['total_nodes_processed'] = total_nodes_processed
        
        print(f"\n{'='*80}")
        print(f"✓ Generation complete!")
        print(f"  Total traces processed: {len(self.traces)}")
        print(f"  Total nodes processed: {total_nodes_processed}")
        print(f"  Candidates per node: {self.n_random + 1} (1 GT + {self.n_random} perturbed)")
        print(f"    - Node perturbations: {self.n_random // 2}")
        print(f"    - Topology perturbations: {self.n_random - self.n_random // 2}")
        print(f"  Total candidates generated: {total_nodes_processed * (self.n_random + 1)}")
        print(f"{'='*80}\n")
        
        # Save to file if output path is provided
        if output_path:
            self._save_json(all_candidates, output_path)
        
        return all_candidates
    
    def print_sample_candidates(self, candidates: Dict, num_samples: int = 3):
        """Print sample candidates for inspection"""
        print(f"\n{'='*80}")
        print(f"Sample Node Candidates (first {num_samples} nodes)")
        print(f"{'='*80}\n")
        
        sample_count = 0
        for trace in candidates['traces']:
            if sample_count >= num_samples:
                break
            
            trace_id = trace['trace_id']
            for node_cand in trace['node_candidates']:
                if sample_count >= num_samples:
                    break
                
                node_id = node_cand['node_id']
                node_task = node_cand['node_task']
                
                print(f"Sample {sample_count + 1}:")
                print(f"  Trace: {trace_id}")
                print(f"  Node: {node_id}")
                print(f"  Task: {node_task}")
                print(f"  Total candidates: {node_cand['total_candidates']}")
                
                # Ground truth info
                gt = node_cand['ground_truth']['tree']
                print(f"  Ground truth tree:")
                print(f"    - Nodes: {len(gt['nodes'])}")
                print(f"    - Edges: {len(gt['edges'])}")
                
                # Random trees info
                print(f"  Random trees: {len(node_cand['random_candidates'])}")
                # Show first 2 of each type if available
                shown = 0
                for i, rand_cand in enumerate(node_cand['random_candidates']):
                    if shown >= 4:  # Show max 4 samples
                        break
                    rt = rand_cand['tree']
                    method = rand_cand.get('perturbation_method', 'unknown')
                    print(f"    Random {i+1} ({method}): {len(rt['nodes'])} nodes, {len(rt['edges'])} edges")
                    shown += 1
                
                print()
                sample_count += 1


def _find_data_file(filename: str) -> str:
    """Smart path resolution to find data files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        filename,
        os.path.join(script_dir, '..', filename),
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
        description='Generate Node Candidates for Multi-Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect tool_pool.json and tool_calling_graphs.json)
  python generate_node_candidates.py
  
  # Specify custom paths
  python generate_node_candidates.py --tool_pool /path/to/tool_pool.json \\
                                     --calling_graph /path/to/graphs.json \\
                                     --output_dir ./output
  
  # Change number of candidate trees
  python generate_node_candidates.py --n_random 20
        """
    )
    
    parser.add_argument(
        '--tool_pool',
        type=str,
        default=None,
        help='Path to tool_pool.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--calling_graph',
        type=str,
        default=None,
        help='Path to tool_calling_graphs.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: ../output)'
    )
    
    parser.add_argument(
        '--n_random',
        type=int,
        default=20,
        help='Number of candidate trees to generate per node via perturbation (default: 20)'
    )
    
    parser.add_argument(
        '--show_samples',
        action='store_true',
        help='Show sample candidates after generation'
    )
    
    args = parser.parse_args()
    
    # Auto-detect data files if not specified
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    
    # Set output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'node_candidates.json')
    
    # Initialize generator
    generator = CallTreeGenerator(
        args.tool_pool,
        args.calling_graph,
        n_random=args.n_random
    )
    
    # Generate candidates
    candidates = generator.generate_candidates_for_all_nodes(output_path)
    
    # Show samples if requested
    if args.show_samples:
        generator.print_sample_candidates(candidates, num_samples=3)


if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    main()

