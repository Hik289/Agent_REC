#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import os
import argparse
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class CallTreeGenerator:
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str, n_random: int = 10):
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
        print(f"  Random trees per node: {n_random}")
        print(f"  Total candidates per node: {n_random + 1} (including ground truth)")
        print(f"\n{'='*80}\n")
    
    def _load_json(self, filepath: str) -> dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_json(self, data: dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved to: {filepath}")
    
    def _build_adjacency_list(self, edges: List[List[str]]) -> Dict[str, List[str]]:
        adj = defaultdict(list)
        for source, target in edges:
            adj[source].append(target)
        return dict(adj)
    
    def _extract_subtree(self, node_id: str, nodes: Dict, adjacency: Dict) -> Dict:
        if node_id not in nodes:
            return {'nodes': [], 'edges': []}
        
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
            
            if current in adjacency:
                for child in adjacency[current]:
                    subtree_edges.append([current, child])
                    queue.append(child)
        
        return {
            'nodes': subtree_nodes,
            'edges': subtree_edges
        }
    
    def _generate_random_tree(self, depth_range: Tuple[int, int] = (1, 5), 
                             branching_range: Tuple[int, int] = (1, 3)) -> Dict:
        depth = random.randint(*depth_range)
        nodes = []
        edges = []
        node_counter = 0
        
        root_tool = random.choice(self.tools)
        nodes.append({
            'node_id': f'rand_{node_counter}',
            'task': f'Call {root_tool}',
            'input_spec': '{}',
            'output_spec': 'result from environment'
        })
        
        current_level = [f'rand_{node_counter}']
        node_counter += 1
        
        for level in range(depth):
            next_level = []
            
            for parent in current_level:
                n_children = random.randint(*branching_range)
                
                for _ in range(n_children):
                    child_tool = random.choice(self.tools)
                    child_id = f'rand_{node_counter}'
                    
                    nodes.append({
                        'node_id': child_id,
                        'task': f'Call {child_tool}',
                        'input_spec': '{}',
                        'output_spec': 'result from environment'
                    })
                    
                    edges.append([parent, child_id])
                    next_level.append(child_id)
                    node_counter += 1
            
            current_level = next_level
            if not current_level:
                break
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def generate_candidates_for_all_nodes(self, output_path: str = None) -> Dict:
        print(f"Generating node candidates...")
        print(f"{'='*80}\n")
        
        all_candidates = {
            'metadata': {
                'n_random_trees': self.n_random,
                'total_candidates_per_node': self.n_random + 1,
                'total_traces': len(self.traces)
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
            
            adjacency = self._build_adjacency_list(edges)
            
            trace_candidates = {
                'trace_id': trace_id,
                'node_candidates': []
            }
            
            for node_id in nodes.keys():
                ground_truth_tree = self._extract_subtree(node_id, nodes, adjacency)
                
                random_trees = []
                for i in range(self.n_random):
                    random_tree = self._generate_random_tree(
                        depth_range=(1, 5),
                        branching_range=(1, 3)
                    )
                    random_trees.append({
                        'candidate_id': f'{node_id}_random_{i}',
                        'type': 'random',
                        'tree': random_tree
                    })
                
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
        print(f"  Candidates per node: {self.n_random + 1}")
        print(f"  Total candidates generated: {total_nodes_processed * (self.n_random + 1)}")
        print(f"{'='*80}\n")
        
        if output_path:
            self._save_json(all_candidates, output_path)
        
        return all_candidates
    
    def print_sample_candidates(self, candidates: Dict, num_samples: int = 3):
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
                
                gt = node_cand['ground_truth']['tree']
                print(f"  Ground truth tree:")
                print(f"    - Nodes: {len(gt['nodes'])}")
                print(f"    - Edges: {len(gt['edges'])}")
                
                print(f"  Random trees: {len(node_cand['random_candidates'])}")
                for i, rand_cand in enumerate(node_cand['random_candidates'][:2]):
                    rt = rand_cand['tree']
                    print(f"    Random {i+1}: {len(rt['nodes'])} nodes, {len(rt['edges'])} edges")
                
                print()
                sample_count += 1


def _find_data_file(filename: str) -> str:
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
    parser = argparse.ArgumentParser(
        description='Generate Node Candidates for Multi-Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_node_candidates.py
  
  python generate_node_candidates.py --tool_pool /path/to/tool_pool.json \\
                                     --calling_graph /path/to/graphs.json \\
                                     --output_dir ./output
  
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
        default=10,
        help='Number of random trees to generate per node (default: 10)'
    )
    
    parser.add_argument(
        '--show_samples',
        action='store_true',
        help='Show sample candidates after generation'
    )
    
    args = parser.parse_args()
    
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'node_candidates.json')
    
    generator = CallTreeGenerator(
        args.tool_pool,
        args.calling_graph,
        n_random=args.n_random
    )
    
    candidates = generator.generate_candidates_for_all_nodes(output_path)
    
    if args.show_samples:
        generator.print_sample_candidates(candidates, num_samples=3)


if __name__ == '__main__':
    random.seed(42)
    main()
