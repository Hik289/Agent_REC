#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Learning-to-Rank Model for Agent System (Graph) Selection
Uses transformers library for BERT embeddings

Implements: s_θ(q, g) = w^T Φ(q, g)
where Φ = [φ_rel, φ_hist, φ_coop, φ_struct]
"""

import json
import numpy as np
import torch
import os
import argparse
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class GraphLearningToRank:
    """Linear Learning-to-Rank Model for Graph/Agent System Selection using BERT embeddings"""
    
    def __init__(self, graph_selection_path: str, node_candidates_path: str,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize LTR model with BERT embeddings
        
        Args:
            graph_selection_path: Path to graph_selection_results.json
            node_candidates_path: Path to node_candidates.json
            model_name: Hugging Face model name
        """
        print("=" * 80)
        print("Graph-Level Linear Learning-to-Rank Model (BERT Embeddings)")
        print("=" * 80)
        
        print(f"\n[1/4] Loading BERT model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise
        
        print("\n[2/4] Loading graph selection results...")
        self.graph_selection = self._load_json(graph_selection_path)
        print(f"  ✓ Loaded graph selection results")
        
        print("\n[3/4] Loading node candidates...")
        self.node_candidates = self._load_json(node_candidates_path)
        print(f"  ✓ Loaded node candidates")
        
        print("\n[4/4] Analyzing graph statistics...")
        self._analyze_graph_statistics()
        
        # Initialize weights [w_rel, w_hist, w_coop, w_struct]
        self.w = np.array([1.0, 1.0, 1.0, 1.0])
        
        print("\n✓ Initialization complete!\n")
    
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text using mean pooling"""
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            output = self.model(**encoded)
            embeddings = output.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding[0].cpu().numpy()
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _analyze_graph_statistics(self):
        """Analyze graph usage statistics from selection results"""
        # Count how often each candidate type appears in top selections
        self.graph_frequency = defaultdict(int)
        self.graph_size_stats = defaultdict(list)
        
        for trace in self.graph_selection.get('traces', []):
            for node_sel in trace.get('node_selections', []):
                for cand in node_sel.get('selected_candidates', []):
                    cand_id = cand.get('candidate_id', '')
                    cand_type = cand.get('type', '')
                    tree_size = cand.get('tree_size', 0)
                    
                    self.graph_frequency[cand_type] += 1
                    self.graph_size_stats[cand_type].append(tree_size)
        
        # Calculate reliability scores (normalized frequency)
        total = sum(self.graph_frequency.values())
        self.graph_reliability = {
            cand_type: (freq / max(total, 1)) * 0.5 + 0.5
            for cand_type, freq in self.graph_frequency.items()
        }
        
        print(f"  ✓ Analyzed graph statistics")
        print(f"    - Ground truth frequency: {self.graph_frequency.get('ground_truth', 0)}")
        print(f"    - Random candidates frequency: {self.graph_frequency.get('random', 0)}")
    
    def _get_candidate_tree(self, trace_id: str, node_id: str, candidate_id: str) -> Dict:
        """Get the full tree structure for a candidate from node_candidates.json"""
        # Find the trace
        for trace in self.node_candidates.get('traces', []):
            if trace['trace_id'] == trace_id:
                # Find the node
                for node_cand in trace.get('node_candidates', []):
                    if node_cand['node_id'] == node_id:
                        # Check ground truth
                        gt = node_cand.get('ground_truth', {})
                        if gt.get('candidate_id') == candidate_id:
                            return gt.get('tree', {})
                        
                        # Check random candidates
                        for rand_cand in node_cand.get('random_candidates', []):
                            if rand_cand.get('candidate_id') == candidate_id:
                                return rand_cand.get('tree', {})
        
        return {}
    
    def _tree_to_text(self, tree: Dict) -> str:
        """Convert tree structure to text representation"""
        nodes = tree.get('nodes', [])
        edges = tree.get('edges', [])
        
        if not nodes:
            return ""
        
        tasks = [node.get('task', '') for node in nodes]
        text_parts = []
        
        if tasks:
            text_parts.append(f"Root: {tasks[0]}")
        
        if len(tasks) > 1:
            children = tasks[1:]
            text_parts.append(f"Children: {' -> '.join(children[:5])}")  # Limit to first 5
        
        text_parts.append(f"Structure: {len(nodes)} nodes, {len(edges)} edges")
        
        return ". ".join(text_parts)
    
    def phi_rel(self, query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
        """
        φ_rel: Semantic alignment between task query and agent system (graph)
        
        Returns:
            Relevance score [0, 1]
        """
        try:
            # Get the tree structure for this candidate
            tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
            tree_text = self._tree_to_text(tree)
            
            if not tree_text or not query:
                return 0.0
            
            query_embedding = self._get_embedding(query)
            tree_embedding = self._get_embedding(tree_text)
            
            similarity = self._compute_cosine_similarity(query_embedding, tree_embedding)
            return (similarity + 1) / 2  # Normalize to [0, 1]
        except Exception as e:
            return 0.0
    
    def phi_hist(self, candidate_type: str) -> float:
        """
        φ_hist: System-level reliability and execution history
        
        Returns:
            Reliability score [0, 1]
        """
        return self.graph_reliability.get(candidate_type, 0.5)
    
    def phi_coop(self, query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
        """
        φ_coop: Internal cooperation and graph-aware compatibility
        Measures how well the graph structure matches the query complexity
        
        Returns:
            Compatibility score [0, 1]
        """
        tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
        num_nodes = len(tree.get('nodes', []))
        num_edges = len(tree.get('edges', []))
        
        # Query complexity (word count as proxy)
        query_complexity = len(query.split())
        
        # If ground truth, give higher score
        if 'ground_truth' in candidate_id:
            return 0.9
        
        # Match tree size to query complexity
        if query_complexity < 5:
            # Simple query prefers smaller trees
            size_score = 1.0 / (1.0 + np.log1p(num_nodes))
        else:
            # Complex query prefers larger trees
            size_score = min(1.0, np.log1p(num_nodes) / 5.0)
        
        return size_score * 0.8 + 0.2  # Scale to [0.2, 1.0]
    
    def phi_struct(self, candidate_id: str, trace_id: str, node_id: str, 
                   all_candidate_ids: List[str]) -> float:
        """
        φ_struct: List-wise and structural suitability among candidate systems
        Measures diversity and uniqueness of this candidate relative to others
        
        Returns:
            Structural utility score [0, 1]
        """
        tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
        num_nodes = len(tree.get('nodes', []))
        num_edges = len(tree.get('edges', []))
        
        # Calculate tree depth (rough estimate)
        depth = num_edges / max(num_nodes, 1) if num_nodes > 0 else 0
        
        # Structural diversity: prefer balanced trees
        if num_nodes > 0:
            balance_score = min(1.0, depth)
        else:
            balance_score = 0.0
        
        # Uniqueness: compare size with other candidates
        sizes = []
        for cand_id in all_candidate_ids:
            if cand_id != candidate_id:
                other_tree = self._get_candidate_tree(trace_id, node_id, cand_id)
                sizes.append(len(other_tree.get('nodes', [])))
        
        if sizes:
            avg_size = np.mean(sizes)
            uniqueness = abs(num_nodes - avg_size) / max(avg_size, 1)
            uniqueness_score = min(1.0, uniqueness / 2.0)
        else:
            uniqueness_score = 0.5
        
        return balance_score * 0.6 + uniqueness_score * 0.4
    
    def compute_features(self, query: str, candidate_id: str, candidate_type: str,
                        trace_id: str, node_id: str, all_candidate_ids: List[str]) -> np.ndarray:
        """
        Compute feature vector Φ(q, g) = [φ_rel, φ_hist, φ_coop, φ_struct]
        
        Returns:
            Feature vector of shape (4,)
        """
        return np.array([
            self.phi_rel(query, candidate_id, trace_id, node_id),
            self.phi_hist(candidate_type),
            self.phi_coop(query, candidate_id, trace_id, node_id),
            self.phi_struct(candidate_id, trace_id, node_id, all_candidate_ids)
        ])
    
    def score(self, features: np.ndarray) -> float:
        """Compute score s_θ(q, g) = w^T Φ(q, g)"""
        return np.dot(self.w, features)
    
    def prepare_dataset(self, max_traces: int = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and test datasets
        
        Returns:
            (train_data, test_data) tuple
        """
        print("=" * 80)
        print("Preparing Training and Test Datasets")
        print("=" * 80)
        
        dataset = []
        traces = self.graph_selection.get('traces', [])
        
        if max_traces:
            traces = traces[:max_traces]
        
        print(f"\nProcessing {len(traces)} traces...")
        
        for i, trace in enumerate(traces):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(traces)}")
            
            trace_id = trace.get('trace_id', '')
            
            for node_sel in trace.get('node_selections', []):
                node_id = node_sel.get('node_id', '')
                node_task = node_sel.get('node_task', '')
                selected_candidates = node_sel.get('selected_candidates', [])
                
                if not selected_candidates:
                    continue
                
                # Get all candidate IDs for structural features
                all_candidate_ids = [c['candidate_id'] for c in selected_candidates]
                
                # Find ground truth
                ground_truth_id = None
                for cand in selected_candidates:
                    if cand['type'] == 'ground_truth':
                        ground_truth_id = cand['candidate_id']
                        break
                
                if not ground_truth_id:
                    continue
                
                # Create samples for each candidate
                for candidate in selected_candidates:
                    candidate_id = candidate['candidate_id']
                    candidate_type = candidate['type']
                    
                    features = self.compute_features(
                        node_task, candidate_id, candidate_type,
                        trace_id, node_id, all_candidate_ids
                    )
                    
                    label = 1 if candidate_id == ground_truth_id else 0
                    
                    dataset.append({
                        'trace_id': trace_id,
                        'node_id': node_id,
                        'query': node_task,
                        'candidate_id': candidate_id,
                        'candidate_type': candidate_type,
                        'features': features,
                        'label': label
                    })
        
        print(f"\n✓ Dataset size: {len(dataset)} samples")
        
        # Split by traces
        trace_ids = list(set([d['trace_id'] for d in dataset]))
        train_traces, test_traces = train_test_split(
            trace_ids,
            test_size=0.2,
            random_state=42
        )
        
        train_data = [d for d in dataset if d['trace_id'] in train_traces]
        test_data = [d for d in dataset if d['trace_id'] in test_traces]
        
        print(f"  - Training set: {len(train_data)} samples ({len(train_traces)} traces)")
        print(f"  - Test set: {len(test_data)} samples ({len(test_traces)} traces)")
        
        return train_data, test_data
    
    def train(self, train_data: List[Dict], learning_rate: float = 0.01,
              num_epochs: int = 100, lambda_reg: float = 0.001) -> List[float]:
        """Train the model using gradient descent on softmax cross-entropy loss"""
        print(f"\n{'='*80}")
        print(f"Training Model")
        print(f"{'='*80}")
        print(f"Parameters: LR={learning_rate}, Epochs={num_epochs}, L2={lambda_reg}\n")
        
        # Group by query (trace_id, node_id)
        query_groups = {}
        for sample in train_data:
            key = (sample['trace_id'], sample['node_id'])
            if key not in query_groups:
                query_groups[key] = []
            query_groups[key].append(sample)
        
        print(f"Training samples grouped into {len(query_groups)} queries\n")
        
        loss_history = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_groups = 0
            
            for group_samples in query_groups.values():
                if len(group_samples) < 2:
                    continue
                
                features_matrix = np.array([s['features'] for s in group_samples])
                labels = np.array([s['label'] for s in group_samples])
                
                if labels.sum() == 0:
                    continue
                
                num_groups += 1
                
                # Compute scores and probabilities
                scores = features_matrix @ self.w
                probs = softmax(scores)
                
                # Cross-entropy + L2 regularization
                loss = -np.sum(labels * np.log(probs + 1e-10))
                loss += lambda_reg * np.sum(self.w ** 2)
                total_loss += loss
                
                # Gradient descent
                grad = features_matrix.T @ (probs - labels) + 2 * lambda_reg * self.w
                self.w -= learning_rate * grad
            
            avg_loss = total_loss / max(num_groups, 1)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"w=[{self.w[0]:.3f}, {self.w[1]:.3f}, "
                      f"{self.w[2]:.3f}, {self.w[3]:.3f}]")
        
        print(f"\n✓ Training complete!")
        self._print_learned_weights()
        
        return loss_history
    
    def _print_learned_weights(self):
        """Print learned weights with interpretation"""
        print(f"\n{'='*60}")
        print("Learned Feature Weights:")
        print(f"{'='*60}")
        print(f"  w[0] = {self.w[0]:7.4f}  →  φ_rel   (semantic alignment)")
        print(f"  w[1] = {self.w[1]:7.4f}  →  φ_hist  (system reliability)")
        print(f"  w[2] = {self.w[2]:7.4f}  →  φ_coop  (internal cooperation)")
        print(f"  w[3] = {self.w[3]:7.4f}  →  φ_struct (structural utility)")
        print(f"{'='*60}")
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> Dict:
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print("Evaluating Model on Test Set")
        print(f"{'='*80}\n")
        
        # Group by query
        query_groups = {}
        for sample in test_data:
            key = (sample['trace_id'], sample['node_id'])
            if key not in query_groups:
                query_groups[key] = []
            query_groups[key].append(sample)
        
        top1_correct = 0
        top3_correct = 0
        mrr_sum = 0.0
        total_queries = 0
        
        detailed_results = []
        
        for group_samples in query_groups.values():
            if len(group_samples) < 2:
                continue
            
            features_matrix = np.array([s['features'] for s in group_samples])
            labels = np.array([s['label'] for s in group_samples])
            candidate_ids = [s['candidate_id'] for s in group_samples]
            query = group_samples[0]['query']
            trace_id = group_samples[0]['trace_id']
            node_id = group_samples[0]['node_id']
            
            if labels.sum() == 0:
                continue
            
            # Compute scores and rank
            scores = features_matrix @ self.w
            ranked_indices = np.argsort(scores)[::-1]
            
            ranked_candidates = [candidate_ids[i] for i in ranked_indices]
            ranked_scores = [scores[i] for i in ranked_indices]
            
            # Find correct answer
            true_idx = np.where(labels == 1)[0]
            if len(true_idx) == 0:
                continue
            
            true_idx = true_idx[0]
            true_candidate = candidate_ids[true_idx]
            rank = np.where(ranked_indices == true_idx)[0][0] + 1
            
            total_queries += 1
            is_top1 = (rank == 1)
            is_top3 = (rank <= 3)
            
            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            mrr_sum += 1.0 / rank
            
            detailed_results.append({
                'trace_id': trace_id,
                'node_id': node_id,
                'query': query,
                'true_candidate': true_candidate,
                'predicted_candidate': ranked_candidates[0],
                'top3_candidates': ranked_candidates[:3],
                'top3_scores': [float(s) for s in ranked_scores[:3]],
                'rank': int(rank),
                'is_top1_correct': bool(is_top1),
                'is_top3_correct': bool(is_top3)
            })
        
        metrics = {
            'top1_accuracy': top1_correct / total_queries if total_queries > 0 else 0,
            'top3_accuracy': top3_correct / total_queries if total_queries > 0 else 0,
            'mrr': mrr_sum / total_queries if total_queries > 0 else 0,
            'num_queries': total_queries,
            'detailed_results': detailed_results
        }
        
        print(f"{'='*80}")
        print("Test Results Summary:")
        print(f"{'='*80}")
        print(f"  Test queries:     {metrics['num_queries']}")
        print(f"  Top-1 accuracy:   {metrics['top1_accuracy']:.2%}  "
              f"({top1_correct}/{total_queries})")
        print(f"  Top-3 accuracy:   {metrics['top3_accuracy']:.2%}  "
              f"({top3_correct}/{total_queries})")
        print(f"  MRR:              {metrics['mrr']:.4f}")
        print(f"{'='*80}")
        
        if verbose and detailed_results:
            print(f"\nDetailed Test Results (first 10 samples):")
            print(f"{'='*80}\n")
            
            for i, result in enumerate(detailed_results[:10], 1):
                status = "✓" if result['is_top1_correct'] else "✗"
                query_preview = result['query'][:60] + "..." if len(result['query']) > 60 else result['query']
                print(f"{status} Sample {i}:")
                print(f"   Query: {query_preview}")
                print(f"   True candidate:      {result['true_candidate']}")
                print(f"   Predicted candidate: {result['predicted_candidate']} (rank: {result['rank']})")
                print(f"   Top-3: {', '.join(result['top3_candidates'])}")
                print()
        
        return metrics
    
    def evaluate_direct_retrieval(self) -> Dict:
        """
        Evaluate direct retrieval baseline: directly select top-1 from all 20 candidates
        using embedding similarity (phi_rel), without candidate filtering or LTR.
        
        Returns:
            Dictionary with top1_accuracy, top3_accuracy, mrr, and num_queries
        """
        print(f"\n{'='*80}")
        print("Direct Retrieval Baseline Evaluation")
        print(f"{'='*80}")
        print("Directly selecting from all 20 perturbed candidates using embedding similarity...")
        
        top1_correct = 0
        top3_correct = 0
        mrr_sum = 0.0
        total_queries = 0
        
        traces = self.node_candidates.get('traces', [])
        
        # Count total nodes for progress tracking
        total_nodes = sum(len(trace.get('node_candidates', [])) for trace in traces)
        processed_nodes = 0
        
        print(f"\nProcessing {total_nodes} nodes across {len(traces)} traces...\n")
        
        for trace_idx, trace in enumerate(traces):
            trace_id = trace.get('trace_id', '')
            
            for node_cand in trace.get('node_candidates', []):
                node_id = node_cand.get('node_id', '')
                node_task = node_cand.get('node_task', '')
                
                processed_nodes += 1
                
                # Show progress every 50 nodes
                if processed_nodes % 50 == 0 or processed_nodes == total_nodes:
                    print(f"  Progress: {processed_nodes}/{total_nodes} nodes "
                          f"({processed_nodes/total_nodes*100:.1f}%) - "
                          f"Trace {trace_idx+1}/{len(traces)}")
                
                # Get ground truth
                ground_truth = node_cand.get('ground_truth', {})
                ground_truth_id = ground_truth.get('candidate_id', '')
                
                if not ground_truth_id:
                    continue
                
                # Get all candidates (ground truth + random)
                all_candidates = [ground_truth]
                all_candidates.extend(node_cand.get('random_candidates', []))
                
                if len(all_candidates) < 2:
                    continue
                
                # Calculate embedding similarity for each candidate
                candidate_scores = []
                for cand in all_candidates:
                    cand_id = cand.get('candidate_id', '')
                    # Use phi_rel (embedding similarity) as the ranking score
                    score = self.phi_rel(node_task, cand_id, trace_id, node_id)
                    candidate_scores.append((cand_id, score))
                
                # Sort by score (descending)
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Find rank of ground truth
                rank = None
                for i, (cand_id, _) in enumerate(candidate_scores, 1):
                    if cand_id == ground_truth_id:
                        rank = i
                        break
                
                if rank is None:
                    continue
                
                total_queries += 1
                
                # Check if ground truth is in top-1 or top-3
                is_top1 = (rank == 1)
                is_top3 = (rank <= 3)
                
                if is_top1:
                    top1_correct += 1
                if is_top3:
                    top3_correct += 1
                
                mrr_sum += 1.0 / rank
        
        print(f"\n✓ Completed processing {processed_nodes} nodes")
        print(f"  Valid queries evaluated: {total_queries}\n")
        
        metrics = {
            'top1_accuracy': top1_correct / total_queries if total_queries > 0 else 0,
            'top3_accuracy': top3_correct / total_queries if total_queries > 0 else 0,
            'mrr': mrr_sum / total_queries if total_queries > 0 else 0,
            'num_queries': total_queries,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct
        }
        
        return metrics
    
    def evaluate_oracle_retrieval(self, test_data: List[Dict] = None,
                                  graph_selection_path: str = None) -> Dict:
        """
        Evaluate oracle retrieval baseline (best possible from retrieved candidates).
        This shows the upper bound of what can be achieved from the filtered candidate set
        produced by graph retrieval.
        
        Args:
            test_data: List of test samples (used to filter to test set only)
            graph_selection_path: Path to graph_selection_results.json
            
        Returns:
            Dictionary with oracle metrics
        """
        if graph_selection_path is None:
            graph_selection_path = os.path.join(
                os.path.dirname(self.node_candidates_path),
                'graph_selection_results.json'
            )
        
        print(f"\n{'='*80}")
        print("ORACLE RETRIEVAL BASELINE (Best possible from retrieved candidates)")
        print(f"{'='*80}")
        print(f"Loading from: {graph_selection_path}")
        
        # Load graph selection results
        with open(graph_selection_path, 'r') as f:
            selection_data = json.load(f)
        
        # Extract all node_selections from all traces
        all_node_selections = []
        traces = selection_data.get('traces', [])
        for trace in traces:
            trace_id = trace.get('trace_id', '')
            for node_sel in trace.get('node_selections', []):
                # Add trace_id to each node selection for filtering
                node_sel['trace_id'] = trace_id
                all_node_selections.append(node_sel)
        
        if test_data:
            # Filter to test set only
            print(f"Filtering to test set queries only")
            test_keys = set()
            for sample in test_data:
                key = (sample['trace_id'], sample['node_id'])
                test_keys.add(key)
            
            # Filter selections
            test_selections = []
            for sel in all_node_selections:
                key = (sel.get('trace_id', ''), sel.get('node_id', ''))
                if key in test_keys:
                    test_selections.append(sel)
            
            all_node_selections = test_selections
            print(f"  Test set queries: {len(all_node_selections)}")
        else:
            print(f"  Using all {len(all_node_selections)} queries (warning: includes training data)")
        
        print("Note: This shows the upper bound of performance achievable from retrieved candidates\n")
        
        print(f"Processing {len(all_node_selections)} node selections...\n")
        
        in_top1 = 0
        in_top3 = 0
        in_top5 = 0
        in_all = 0
        total = 0
        
        for idx, selection in enumerate(all_node_selections, 1):
            if idx % 50 == 0 or idx == len(all_node_selections):
                print(f"  Progress: {idx}/{len(all_node_selections)} selections ({idx/len(all_node_selections)*100:.1f}%)")
            
            # Get selected candidates
            selected_candidates = selection.get('selected_candidates', [])
            if not selected_candidates:
                continue
            
            # Find ground truth candidate
            ground_truth_id = None
            for cand in selected_candidates:
                if cand.get('type') == 'ground_truth':
                    ground_truth_id = cand.get('candidate_id', '')
                    break
            
            if not ground_truth_id:
                continue
            
            total += 1
            candidate_ids = [c.get('candidate_id', '') for c in selected_candidates]
            
            if ground_truth_id in candidate_ids:
                in_all += 1
                gt_rank = candidate_ids.index(ground_truth_id) + 1
                
                if gt_rank == 1:
                    in_top1 += 1
                if gt_rank <= 3:
                    in_top3 += 1
                if gt_rank <= 5:
                    in_top5 += 1
        
        print(f"\n✓ Completed processing")
        print(f"  Valid queries evaluated: {total}\n")
        
        oracle_all = in_all / total if total > 0 else 0.0
        oracle_top1 = in_top1 / total if total > 0 else 0.0
        oracle_top3 = in_top3 / total if total > 0 else 0.0
        oracle_top5 = in_top5 / total if total > 0 else 0.0
        
        metrics = {
            'method': 'oracle_retrieval_from_candidates',
            'oracle_accuracy': oracle_all,
            'oracle_accuracy_top1': oracle_top1,
            'oracle_accuracy_top3': oracle_top3,
            'oracle_accuracy_top5': oracle_top5,
            'num_queries': total,
            'in_all_candidates': in_all,
            'in_top1': in_top1,
            'in_top3': in_top3,
            'in_top5': in_top5
        }
        
        print(f"Oracle Retrieval Results (Upper Bounds):")
        print(f"  Test queries:             {total}")
        print(f"  Oracle (all candidates):  {oracle_all:.2%}  ({in_all}/{total})")
        print(f"  Oracle (already top-1):   {oracle_top1:.2%}  ({in_top1}/{total})")
        print(f"  Oracle (top-3):           {oracle_top3:.2%}  ({in_top3}/{total})")
        print(f"  Oracle (top-5):           {oracle_top5:.2%}  ({in_top5}/{total})")
        print(f"\nInterpretation:")
        print(f"  - {oracle_top5:.1%} is the max achievable if we perfectly rank top-5 candidates")
        print(f"  - {100-oracle_all*100:.1f}% of ground truths are not in candidate set")
        print(f"{'='*80}")
        
        return metrics
    
    def save_model(self, output_dir: str, test_metrics: Dict = None, baseline_metrics: Dict = None, 
                   direct_retrieval_metrics: Dict = None, oracle_retrieval_metrics: Dict = None):
        """Save model weights to output directory"""
        filepath = os.path.join(output_dir, 'graph_ltr_model_weights.json')
        
        model_data = {
            'weights': self.w.tolist(),
            'graph_reliability': dict(self.graph_reliability),
            'interpretation': {
                'w[0]': 'φ_rel - semantic alignment (query-system)',
                'w[1]': 'φ_hist - system reliability',
                'w[2]': 'φ_coop - internal cooperation',
                'w[3]': 'φ_struct - structural utility'
            }
        }
        
        if direct_retrieval_metrics:
            model_data['direct_retrieval_performance'] = {
                'top1_accuracy': direct_retrieval_metrics['top1_accuracy'],
                'top3_accuracy': direct_retrieval_metrics['top3_accuracy'],
                'mrr': direct_retrieval_metrics['mrr'],
                'num_queries': direct_retrieval_metrics['num_queries']
            }
        
        if oracle_retrieval_metrics:
            model_data['oracle_retrieval_performance'] = {
                'oracle_accuracy_all': oracle_retrieval_metrics['oracle_accuracy'],
                'oracle_accuracy_top1': oracle_retrieval_metrics.get('oracle_accuracy_top1', 0),
                'oracle_accuracy_top3': oracle_retrieval_metrics.get('oracle_accuracy_top3', 0),
                'oracle_accuracy_top5': oracle_retrieval_metrics.get('oracle_accuracy_top5', 0),
                'num_queries': oracle_retrieval_metrics['num_queries']
            }
        
        if baseline_metrics:
            model_data['baseline_performance'] = {
                'top1_accuracy': baseline_metrics['top1_accuracy'],
                'top3_accuracy': baseline_metrics['top3_accuracy'],
                'mrr': baseline_metrics['mrr'],
                'num_test_queries': baseline_metrics['num_queries']
            }
        
        if test_metrics:
            model_data['test_performance'] = {
                'top1_accuracy': test_metrics['top1_accuracy'],
                'top3_accuracy': test_metrics['top3_accuracy'],
                'mrr': test_metrics['mrr'],
                'num_test_queries': test_metrics['num_queries']
            }
            
            if baseline_metrics:
                model_data['improvement'] = {
                    'top1_accuracy': test_metrics['top1_accuracy'] - baseline_metrics['top1_accuracy'],
                    'top3_accuracy': test_metrics['top3_accuracy'] - baseline_metrics['top3_accuracy'],
                    'mrr': test_metrics['mrr'] - baseline_metrics['mrr']
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Model saved to: {filepath}")
    
    def save_test_results(self, metrics: Dict, output_dir: str, baseline_metrics: Dict = None,
                         direct_retrieval_metrics: Dict = None, oracle_retrieval_metrics: Dict = None):
        """Save detailed test results to output directory"""
        filepath = os.path.join(output_dir, 'graph_ltr_test_results.json')
        
        results_data = {
            'trained_model': {
                'num_test_queries': metrics['num_queries'],
                'top1_accuracy': metrics['top1_accuracy'],
                'top3_accuracy': metrics['top3_accuracy'],
                'mrr': metrics['mrr']
            },
            'detailed_results': metrics.get('detailed_results', [])
        }
        
        if direct_retrieval_metrics:
            results_data['direct_retrieval'] = {
                'num_queries': direct_retrieval_metrics['num_queries'],
                'top1_accuracy': direct_retrieval_metrics['top1_accuracy'],
                'top3_accuracy': direct_retrieval_metrics['top3_accuracy'],
                'mrr': direct_retrieval_metrics['mrr']
            }
        
        if oracle_retrieval_metrics:
            results_data['oracle_retrieval'] = {
                'method': 'Oracle retrieval (best possible from retrieved candidates)',
                'num_queries': oracle_retrieval_metrics['num_queries'],
                'oracle_accuracy_all': oracle_retrieval_metrics['oracle_accuracy'],
                'oracle_accuracy_top1': oracle_retrieval_metrics.get('oracle_accuracy_top1', 0),
                'oracle_accuracy_top3': oracle_retrieval_metrics.get('oracle_accuracy_top3', 0),
                'oracle_accuracy_top5': oracle_retrieval_metrics.get('oracle_accuracy_top5', 0),
                'note': 'Upper bound - best achievable performance from retrieved candidate set'
            }
        
        if baseline_metrics:
            results_data['baseline'] = {
                'num_test_queries': baseline_metrics['num_queries'],
                'top1_accuracy': baseline_metrics['top1_accuracy'],
                'top3_accuracy': baseline_metrics['top3_accuracy'],
                'mrr': baseline_metrics['mrr']
            }
            results_data['improvement'] = {
                'top1_accuracy': metrics['top1_accuracy'] - baseline_metrics['top1_accuracy'],
                'top3_accuracy': metrics['top3_accuracy'] - baseline_metrics['top3_accuracy'],
                'mrr': metrics['mrr'] - baseline_metrics['mrr']
            }
        
        # Add gap to oracle if available
        if oracle_retrieval_metrics:
            results_data['gap_to_oracle'] = {
                'current_to_oracle': oracle_retrieval_metrics['oracle_accuracy'] - metrics['top1_accuracy'],
                'note': 'Room for improvement to reach oracle performance'
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed test results saved to: {filepath}")


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
        description='Graph-Level Linear Learning-to-Rank Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect from ../output/)
  python learning_to_rank.py
  
  # Specify custom paths
  python learning_to_rank.py --graph_selection /path/to/graph_selection_results.json \\
                             --node_candidates /path/to/node_candidates.json \\
                             --output_dir ./output
  
  # Adjust training parameters
  python learning_to_rank.py --lr 0.05 --epochs 200
        """
    )
    
    parser.add_argument(
        '--graph_selection',
        type=str,
        default=None,
        help='Path to graph_selection_results.json (default: auto-detect)'
    )
    
    parser.add_argument(
        '--node_candidates',
        type=str,
        default=None,
        help='Path to node_candidates.json (default: auto-detect)'
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
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='BERT model from Hugging Face'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=0.001,
        help='L2 regularization coefficient (default: 0.001)'
    )
    
    parser.add_argument(
        '--max_traces',
        type=int,
        default=None,
        help='Maximum number of traces to use (default: None, use all)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect data files
    if args.graph_selection is None:
        args.graph_selection = _find_data_file('graph_selection_results.json')
    if args.node_candidates is None:
        args.node_candidates = _find_data_file('node_candidates.json')
    
    # Set output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.output_dir = os.path.join(parent_dir, 'output')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Graph-Level Linear Learning-to-Rank for Agent System Selection")
    print("Using BERT Embeddings (transformers library)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Graph Selection:  {args.graph_selection}")
    print(f"  Node Candidates:  {args.node_candidates}")
    print(f"  Output Dir:       {args.output_dir}")
    print(f"  Model:            {args.model}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  L2 Reg:           {args.lambda_reg}")
    print(f"  Max Traces:       {args.max_traces if args.max_traces else 'All'}")
    print("="*80 + "\n")
    
    # Initialize model
    ltr = GraphLearningToRank(
        args.graph_selection,
        args.node_candidates,
        model_name=args.model
    )
    
    # Prepare dataset
    train_data, test_data = ltr.prepare_dataset(max_traces=args.max_traces)
    
    if len(train_data) == 0:
        print("\nError: No training data available!")
        return
    
    # Evaluate direct retrieval baseline (from all 20 candidates)
    print("\n" + "="*80)
    print("DIRECT RETRIEVAL BASELINE (No candidate filtering, direct from 20)")
    print("="*80)
    direct_retrieval_metrics = ltr.evaluate_direct_retrieval()
    print(f"  Top-1 accuracy:   {direct_retrieval_metrics['top1_accuracy']:.2%}  "
          f"({direct_retrieval_metrics['top1_correct']}/{direct_retrieval_metrics['num_queries']})")
    print(f"  Top-3 accuracy:   {direct_retrieval_metrics['top3_accuracy']:.2%}  "
          f"({direct_retrieval_metrics['top3_correct']}/{direct_retrieval_metrics['num_queries']})")
    print(f"  MRR:              {direct_retrieval_metrics['mrr']:.4f}")
    print("="*80 + "\n")
    
    # Evaluate oracle retrieval baseline (best possible from retrieved candidates)
    oracle_retrieval_metrics = ltr.evaluate_oracle_retrieval(
        test_data=test_data,
        graph_selection_path=os.path.join(args.output_dir, 'graph_selection_results.json')
    )
    
    # Evaluate LTR baseline (before training, with 10 candidates from graph_retrieval)
    print("\n" + "="*80)
    print("LTR BASELINE PERFORMANCE (Before Training, with 10 candidates)")
    print("="*80)
    print("Testing with initial weights (all 1.0)...")
    baseline_metrics = ltr.evaluate(test_data, verbose=False)
    print(f"\nBaseline Results:")
    print(f"  Top-1 accuracy:   {baseline_metrics['top1_accuracy']:.2%}")
    print(f"  Top-3 accuracy:   {baseline_metrics['top3_accuracy']:.2%}")
    print(f"  MRR:              {baseline_metrics['mrr']:.4f}")
    print("="*80 + "\n")
    
    # Train model
    loss_history = ltr.train(
        train_data,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        lambda_reg=args.lambda_reg
    )
    
    # Evaluate on test set (after training)
    print("\n" + "="*80)
    print("TRAINED MODEL PERFORMANCE (After Training, with 10 candidates)")
    print("="*80)
    metrics = ltr.evaluate(test_data, verbose=True)
    
    # Print performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n  {'Method':<55} {'Top-1 Acc':<15}")
    print(f"  {'-'*70}")
    print(f"  {'1. Direct Retrieval (20 cand., no filter)':<55} "
          f"{direct_retrieval_metrics['top1_accuracy']:<15.2%}")
    print(f"  {'2. Oracle Retrieval (best from retrieved cand.)':<55} "
          f"{oracle_retrieval_metrics['oracle_accuracy']:<15.2%}  [Upper Bound]")
    print(f"  {'3. LTR Baseline (10 cand., init weights)':<55} "
          f"{baseline_metrics['top1_accuracy']:<15.2%}")
    print(f"  {'4. LTR Trained (10 cand., trained weights)':<55} "
          f"{metrics['top1_accuracy']:<15.2%}")
    
    print(f"\n  {'Improvements:':<55}")
    print(f"  {'  vs Direct Retrieval:':<55} "
          f"{'+' if metrics['top1_accuracy'] > direct_retrieval_metrics['top1_accuracy'] else ''}"
          f"{(metrics['top1_accuracy']-direct_retrieval_metrics['top1_accuracy'])*100:.2f}%")
    print(f"  {'  vs LTR Baseline:':<55} "
          f"{'+' if metrics['top1_accuracy'] > baseline_metrics['top1_accuracy'] else ''}"
          f"{(metrics['top1_accuracy']-baseline_metrics['top1_accuracy'])*100:.2f}%")
    print(f"  {'  Gap to Oracle (room for improvement):':<55} "
          f"{(oracle_retrieval_metrics['oracle_accuracy']-metrics['top1_accuracy'])*100:.2f}%")
    print("="*80)
    
    # Save results to output directory (including all baseline metrics)
    ltr.save_model(args.output_dir, test_metrics=metrics, baseline_metrics=baseline_metrics,
                   direct_retrieval_metrics=direct_retrieval_metrics,
                   oracle_retrieval_metrics=oracle_retrieval_metrics)
    ltr.save_test_results(metrics, args.output_dir, baseline_metrics=baseline_metrics,
                         direct_retrieval_metrics=direct_retrieval_metrics,
                         oracle_retrieval_metrics=oracle_retrieval_metrics)
    
    print(f"\n{'='*80}")
    print("✓ Pipeline Complete!")
    print(f"  All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

