#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Learning-to-Rank Model for Tool Selection
Uses transformers library directly (more stable on Mac than sentence-transformers)

Implements: s_θ(q, a) = w^T Φ(q, a)
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
import warnings
warnings.filterwarnings('ignore')


class LinearLearningToRank:
    """Linear Learning-to-Rank Model using BERT embeddings"""
    
    def __init__(self, tool_pool_path: str, calling_graph_path: str,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize LTR model with BERT embeddings
        
        Args:
            tool_pool_path: Path to tool_pool.json
            calling_graph_path: Path to tool_calling_graphs.json
            model_name: Hugging Face model name
        """
        print("=" * 80)
        print("Initializing Linear Learning-to-Rank Model (BERT Embeddings)")
        print("=" * 80)
        
        print(f"\n[1/4] Loading BERT model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            print("  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise
        
        print("\n[2/4] Loading data...")
        self.tool_pool = self._load_json(tool_pool_path)
        self.tools = self.tool_pool.get('tools', {})
        print(f"  ✓ Loaded {len(self.tools)} tools")
        
        print("\n[3/4] Precomputing tool embeddings...")
        self._precompute_tool_embeddings()
        
        print("\n[4/4] Analyzing tool usage statistics...")
        self._analyze_tool_statistics(calling_graph_path)
        
        # Initialize weights [w_rel, w_hist, w_coop, w_struct]
        self.w = np.array([1.0, 1.0, 1.0, 1.0])
        
        print("\n✓ Initialization complete!\n")
    
    def _load_json(self, filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for text using mean pooling
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Get model output
            output = self.model(**encoded)
            
            # Mean pooling
            embeddings = output.last_hidden_state
            attention_mask = encoded['attention_mask']
            
            # Mask and mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding[0].cpu().numpy()
    
    def _precompute_tool_embeddings(self):
        """Precompute embeddings for all tools"""
        self.tool_names = []
        self.tool_embeddings_list = []
        
        for tool_name, tool_info in self.tools.items():
            self.tool_names.append(tool_name)
            
            # Create tool description
            desc = tool_info.get('description', '')
            inputs = tool_info.get('inputs', '')
            tool_text = f"Tool: {tool_name}. Description: {desc}"
            if inputs and inputs != "no parameters":
                tool_text += f" Inputs: {inputs}"
            
            # Get embedding
            embedding = self._get_embedding(tool_text)
            self.tool_embeddings_list.append(embedding)
        
        # Stack into matrix
        self.tool_embeddings = np.vstack(self.tool_embeddings_list)
        
        self.tool_name_to_idx = {
            name: idx for idx, name in enumerate(self.tool_names)
        }
        
        print(f"  ✓ Computed embeddings for {len(self.tool_names)} tools")
        print(f"    Embedding dimension: {self.tool_embeddings.shape[1]}")
    
    def _analyze_tool_statistics(self, filepath: str):
        """Analyze tool usage statistics from calling graph"""
        data = self._load_json(filepath)
        traces = data.get('traces', [])
        
        tool_freq = {tool: 0 for tool in self.tools.keys()}
        
        # Process subset for efficiency
        max_traces_for_stats = min(30, len(traces))
        for trace in traces[:max_traces_for_stats]:
            nodes = trace.get('nodes', {})
            for node_info in nodes.values():
                task = node_info.get('task', '')
                if task.startswith('Call '):
                    tool_name = task.replace('Call ', '').strip()
                    if tool_name in tool_freq:
                        tool_freq[tool_name] += 1
        
        # Normalize frequency as reliability score
        total = sum(tool_freq.values())
        self.tool_reliability = {
            tool: (freq / max(total, 1)) * 0.5 + 0.5
            for tool, freq in tool_freq.items()
        }
        
        print(f"  ✓ Analyzed {max_traces_for_stats} traces for tool usage statistics")
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def phi_rel(self, query: str, tool_name: str) -> float:
        """
        φ_rel: Tool-query relevance (BERT embedding similarity)
        
        Returns:
            Relevance score [0, 1]
        """
        if tool_name not in self.tool_name_to_idx:
            return 0.0
        
        try:
            query_embedding = self._get_embedding(query)
            tool_idx = self.tool_name_to_idx[tool_name]
            tool_embedding = self.tool_embeddings[tool_idx]
            
            similarity = self._compute_cosine_similarity(query_embedding, tool_embedding)
            # Normalize to [0, 1]
            return (similarity + 1) / 2
        except Exception as e:
            print(f"Warning: Similarity calculation error: {e}")
            return 0.0
    
    def phi_hist(self, tool_name: str) -> float:
        """
        φ_hist: Historical reliability prior
        
        Returns:
            Reliability score [0, 1]
        """
        return self.tool_reliability.get(tool_name, 0.5)
    
    def phi_coop(self, query: str, tool_name: str) -> float:
        """
        φ_coop: Graph-aware compatibility (name matching)
        
        Returns:
            Compatibility score [0, 1]
        """
        tool_name_lower = tool_name.lower()
        query_lower = query.lower()
        
        if tool_name_lower in query_lower:
            return 1.0
        elif any(word in query_lower for word in tool_name_lower.split('_')):
            return 0.7
        else:
            return 0.3
    
    def phi_struct(self, query: str, tool_name: str) -> float:
        """
        φ_struct: Structural utility (relevance transformation)
        
        Returns:
            Structural utility score [0, 1]
        """
        phi_rel_val = self.phi_rel(query, tool_name)
        return np.sqrt(phi_rel_val + 0.1)
    
    def compute_features(self, query: str, tool_name: str) -> np.ndarray:
        """
        Compute feature vector Φ(q, a) = [φ_rel, φ_hist, φ_coop, φ_struct]
        
        Returns:
            Feature vector of shape (4,)
        """
        if tool_name not in self.tool_name_to_idx:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        return np.array([
            self.phi_rel(query, tool_name),
            self.phi_hist(tool_name),
            self.phi_coop(query, tool_name),
            self.phi_struct(query, tool_name)
        ])
    
    def score(self, query: str, tool_name: str) -> float:
        """
        Compute score s_θ(q, a) = w^T Φ(q, a)
        
        Returns:
            Weighted score
        """
        features = self.compute_features(query, tool_name)
        return np.dot(self.w, features)
    
    def prepare_dataset(self, calling_graph_path: str = None, 
                       tool_selection_path: str = None,
                       max_traces: int = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and test datasets using tool_selection results
        
        Args:
            calling_graph_path: Path to calling graph JSON (for compatibility, not used)
            tool_selection_path: Path to tool_selection_results.json
            max_traces: Maximum number of traces to process (None = all)
            
        Returns:
            (train_data, test_data) tuple
        """
        if tool_selection_path is None:
            tool_selection_path = _find_data_file('tool_selection_results.json')
        
        print("=" * 80)
        print("Preparing Training and Test Datasets")
        print("=" * 80)
        print(f"Loading tool selection results from: {tool_selection_path}")
        
        # Load tool selection results
        selection_data = self._load_json(tool_selection_path)
        all_selections = selection_data.get('selections', [])
        selections = all_selections if max_traces is None else all_selections[:max_traces]
        
        dataset = []
        
        print(f"\nProcessing {len(selections)} decision points...")
        for i, selection in enumerate(selections):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(selections)}")
            
            trace_id = selection.get('trace_id', '')
            node_id = selection.get('node_id', '')
            query = selection.get('task', '')
            ground_truth = selection.get('ground_truth', '')
            
            # Support both old and new data formats
            # New format: retrieved_candidates (list of 5 candidates)
            # Old format: selected_tools (list of 3-4 tools)
            candidates = selection.get('retrieved_candidates', selection.get('selected_tools', []))
            
            # Skip if no ground truth or no candidates
            if not ground_truth or not candidates:
                continue
            
            # Skip if ground truth not in tool pool
            if ground_truth not in self.tools:
                continue
            
            # Use the retrieved candidates as training samples
            for tool_info in candidates:
                candidate = tool_info.get('tool_name', '')
                
                if candidate not in self.tools:
                    continue
                
                features = self.compute_features(query, candidate)
                label = 1 if candidate == ground_truth else 0
                
                dataset.append({
                    'trace_id': trace_id,
                    'node_id': node_id,
                    'query': query,
                    'tool': candidate,
                    'features': features,
                    'label': label
                })
        
        print(f"\n✓ Dataset size: {len(dataset)} samples")
        
        # Check if dataset is empty or too small
        if len(dataset) == 0:
            print("\n" + "="*80)
            print("ERROR: No valid samples found in the dataset!")
            print("="*80)
            print("\nPossible reasons:")
            print("  1. The calling graph doesn't contain ground truth labels")
            print("     - Check if nodes have tasks starting with 'Call [tool_name]'")
            print("  2. Tool names in the calling graph don't match tool_pool.json")
            print("  3. No valid decision points with candidates found")
            print("\nSuggestions:")
            print("  - Verify the format of your calling graph matches the expected structure")
            print("  - Check that tool names are consistent across files")
            print("  - Ensure nodes have proper 'task' fields")
            print("="*80)
            raise ValueError("Dataset preparation failed: No valid samples found. See error message above.")
        
        # Check if we have enough traces for splitting
        trace_ids = list(set([d['trace_id'] for d in dataset]))
        
        if len(trace_ids) < 2:
            print(f"\n⚠ Warning: Only {len(trace_ids)} trace(s) found. Using all data for both train and test.")
            return dataset, dataset
        
        # Adjust test_size if we have very few traces
        if len(trace_ids) < 5:
            test_size = 1  # Use 1 trace for testing
            print(f"\n⚠ Warning: Only {len(trace_ids)} traces. Using 1 trace for test, rest for training.")
            train_traces = trace_ids[:-1]
            test_traces = [trace_ids[-1]]
        else:
            # Normal split by traces to avoid data leakage
            test_size = max(0.2, 1.0 / len(trace_ids))  # At least one trace for testing
            train_traces, test_traces = train_test_split(
                trace_ids, 
                test_size=test_size, 
                random_state=42
            )
        
        train_data = [d for d in dataset if d['trace_id'] in train_traces]
        test_data = [d for d in dataset if d['trace_id'] in test_traces]
        
        print(f"  - Training set: {len(train_data)} samples ({len(train_traces)} traces)")
        print(f"  - Test set: {len(test_data)} samples ({len(test_traces)} traces)")
        
        return train_data, test_data
    
    def train(self, train_data: List[Dict], learning_rate: float = 0.01,
              num_epochs: int = 100, lambda_reg: float = 0.001) -> List[float]:
        """
        Train the model using gradient descent on softmax cross-entropy loss
        
        Args:
            train_data: Training dataset
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            lambda_reg: L2 regularization coefficient
            
        Returns:
            Loss history list
        """
        print(f"\n{'='*80}")
        print(f"Training Model")
        print(f"{'='*80}")
        print(f"Parameters: LR={learning_rate}, Epochs={num_epochs}, L2={lambda_reg}\n")
        
        # Group by query
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
        print(f"  w[0] = {self.w[0]:7.4f}  →  φ_rel   (tool-query relevance)")
        print(f"  w[1] = {self.w[1]:7.4f}  →  φ_hist  (historical reliability)")
        print(f"  w[2] = {self.w[2]:7.4f}  →  φ_coop  (graph-aware compatibility)")
        print(f"  w[3] = {self.w[3]:7.4f}  →  φ_struct (structural utility)")
        print(f"{'='*60}")
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> Dict[str, any]:
        """
        Evaluate model on test set
        
        Args:
            test_data: Test dataset
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with metrics and detailed results
        """
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
            tools = [s['tool'] for s in group_samples]
            query = group_samples[0]['query']
            trace_id = group_samples[0]['trace_id']
            node_id = group_samples[0]['node_id']
            
            if labels.sum() == 0:
                continue
            
            # Compute scores and rank
            scores = features_matrix @ self.w
            ranked_indices = np.argsort(scores)[::-1]
            
            # Get ranked tools
            ranked_tools = [tools[i] for i in ranked_indices]
            ranked_scores = [scores[i] for i in ranked_indices]
            
            # Find correct answer
            true_idx = np.where(labels == 1)[0]
            if len(true_idx) == 0:
                continue
            
            true_idx = true_idx[0]
            true_tool = tools[true_idx]
            rank = np.where(ranked_indices == true_idx)[0][0] + 1
            
            total_queries += 1
            is_top1 = (rank == 1)
            is_top3 = (rank <= 3)
            
            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            mrr_sum += 1.0 / rank
            
            # Record detailed results
            detailed_results.append({
                'trace_id': trace_id,
                'node_id': node_id,
                'query': query,
                'true_tool': true_tool,
                'predicted_tool': ranked_tools[0],
                'top3_tools': ranked_tools[:3],
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
        
        # Print summary
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
                query_preview = result['query'][:80] + "..." if len(result['query']) > 80 else result['query']
                print(f"{status} Sample {i}:")
                print(f"   Query: {query_preview}")
                print(f"   True tool:      {result['true_tool']}")
                print(f"   Predicted tool: {result['predicted_tool']} (rank: {result['rank']})")
                print(f"   Top-3: {', '.join(result['top3_tools'])}")
                print()
        
        return metrics
    
    def evaluate_direct_retrieval(self, tool_selection_path: str = None) -> Dict:
        """
        Evaluate direct retrieval baseline (top-1 from all tools without LTR)
        
        Args:
            tool_selection_path: Path to tool_selection_results.json
            
        Returns:
            Dictionary with metrics
        """
        if tool_selection_path is None:
            tool_selection_path = _find_data_file('tool_selection_results.json')
        
        print("\n" + "="*80)
        print("DIRECT RETRIEVAL BASELINE (No candidate selection, direct top-1)")
        print("="*80)
        print(f"Loading from: {tool_selection_path}")
        
        # Load selection results
        selection_data = self._load_json(tool_selection_path)
        selections = selection_data.get('selections', [])
        
        print(f"\nProcessing {len(selections)} tool selection decisions...\n")
        
        correct = 0
        total = 0
        
        for idx, selection in enumerate(selections, 1):
            # Show progress every 100 selections
            if idx % 100 == 0 or idx == len(selections):
                print(f"  Progress: {idx}/{len(selections)} decisions ({idx/len(selections)*100:.1f}%)")
            ground_truth = selection.get('ground_truth', '')
            if not ground_truth:
                continue
            
            # Get the selected tool (which is top-1 from direct retrieval)
            selected_tool_info = selection.get('selected_tool')
            if not selected_tool_info:
                continue
            
            selected_tool = selected_tool_info.get('tool_name', '')
            
            total += 1
            if selected_tool == ground_truth:
                correct += 1
        
        print(f"\n✓ Completed processing {len(selections)} decisions")
        print(f"  Valid queries evaluated: {total}\n")
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            'method': 'direct_retrieval',
            'top1_accuracy': accuracy,
            'num_queries': total,
            'correct': correct
        }
        
        print(f"Direct Retrieval Results:")
        print(f"  Test queries:     {total}")
        print(f"  Top-1 accuracy:   {accuracy:.2%}  ({correct}/{total})")
        print("="*80)
        
        return metrics
    
    def evaluate_oracle_retrieval(self, test_data: List[Dict] = None, 
                                  tool_selection_path: str = None) -> Dict:
        """
        Evaluate oracle retrieval baseline (best possible from candidates)
        This shows the upper bound of what can be achieved from the retrieved candidate set.
        
        Args:
            test_data: List of test data samples (used to identify which traces are in test set)
            tool_selection_path: Path to tool_selection_results.json
            
        Returns:
            Dictionary with metrics
        """
        if tool_selection_path is None:
            tool_selection_path = _find_data_file('tool_selection_results.json')
        
        print("\n" + "="*80)
        print("ORACLE RETRIEVAL BASELINE (Best possible from candidate set)")
        print("="*80)
        print(f"Loading from: {tool_selection_path}")
        
        # Load selection results
        selection_data = self._load_json(tool_selection_path)
        selections = selection_data.get('selections', [])
        
        if test_data:
            # Filter selections to only test set traces/nodes
            print(f"Filtering to test set queries only")
            test_keys = set()
            for sample in test_data:
                key = (sample['trace_id'], sample['node_id'])
                test_keys.add(key)
            
            # Filter selections
            test_selections = []
            for sel in selections:
                key = (sel.get('trace_id', ''), sel.get('node_id', ''))
                if key in test_keys:
                    test_selections.append(sel)
            
            # Get unique queries (dedup by trace+node)
            seen = set()
            unique_selections = []
            for sel in test_selections:
                key = (sel.get('trace_id', ''), sel.get('node_id', ''))
                if key not in seen:
                    seen.add(key)
                    unique_selections.append(sel)
            
            selections = unique_selections
            print(f"  Test set queries: {len(selections)}")
        else:
            print(f"  Using all {len(selections)} queries (warning: includes training data)")
        
        print("Note: This shows the upper bound of performance achievable from candidates\n")
        
        print(f"Processing {len(selections)} tool selection decisions...\n")
        
        in_top1 = 0
        in_top3 = 0
        in_top5 = 0
        in_all = 0
        total = 0
        
        for idx, selection in enumerate(selections, 1):
            if idx % 50 == 0 or idx == len(selections):
                print(f"  Progress: {idx}/{len(selections)} decisions ({idx/len(selections)*100:.1f}%)")
            
            ground_truth = selection.get('ground_truth', '')
            if not ground_truth:
                continue
            
            # Get all retrieved candidates
            candidates = selection.get('retrieved_candidates', [])
            if not candidates:
                continue
            
            total += 1
            candidate_names = [c.get('tool_name', '') for c in candidates]
            
            if ground_truth in candidate_names:
                in_all += 1
                gt_rank = candidate_names.index(ground_truth) + 1
                
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
            'oracle_accuracy': oracle_all,      # Best possible from all candidates
            'oracle_accuracy_top1': oracle_top1,  # GT is already top-1
            'oracle_accuracy_top3': oracle_top3,  # Best possible from top-3
            'oracle_accuracy_top5': oracle_top5,  # Best possible from top-5
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
        print("="*80)
        
        return metrics
    
    def save_model(self, output_dir: str, test_metrics: Dict = None, baseline_metrics: Dict = None,
                   direct_retrieval_metrics: Dict = None, oracle_retrieval_metrics: Dict = None):
        """Save model weights to output directory"""
        filepath = os.path.join(output_dir, 'ltr_model_weights.json')
        
        model_data = {
            'weights': self.w.tolist(),
            'tool_reliability': self.tool_reliability,
            'interpretation': {
                'w[0]': 'φ_rel - tool-query relevance',
                'w[1]': 'φ_hist - historical reliability',
                'w[2]': 'φ_coop - graph-aware compatibility',
                'w[3]': 'φ_struct - structural utility'
            }
        }
        
        if direct_retrieval_metrics:
            model_data['direct_retrieval_performance'] = {
                'method': 'direct_retrieval (no candidate selection)',
                'top1_accuracy': direct_retrieval_metrics['top1_accuracy'],
                'num_test_queries': direct_retrieval_metrics['num_queries']
            }
        
        if baseline_metrics:
            model_data['baseline_performance'] = {
                'method': 'LTR with initial weights on 5 candidates',
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
        filepath = os.path.join(output_dir, 'ltr_test_results.json')
        
        results_data = {
            'trained_model': {
                'method': 'LTR (trained) on 5 candidates',
                'num_test_queries': metrics['num_queries'],
                'top1_accuracy': metrics['top1_accuracy'],
                'top3_accuracy': metrics['top3_accuracy'],
                'mrr': metrics['mrr']
            },
            'detailed_results': metrics.get('detailed_results', [])
        }
        
        if direct_retrieval_metrics:
            results_data['direct_retrieval'] = {
                'method': 'Direct retrieval (no candidate selection)',
                'num_test_queries': direct_retrieval_metrics['num_queries'],
                'top1_accuracy': direct_retrieval_metrics['top1_accuracy']
            }
        
        if oracle_retrieval_metrics:
            results_data['oracle_retrieval'] = {
                'method': 'Oracle retrieval (best possible from candidates)',
                'num_test_queries': oracle_retrieval_metrics['num_queries'],
                'oracle_accuracy_all': oracle_retrieval_metrics['oracle_accuracy'],
                'oracle_accuracy_top1': oracle_retrieval_metrics.get('oracle_accuracy_top1', 0),
                'oracle_accuracy_top3': oracle_retrieval_metrics.get('oracle_accuracy_top3', oracle_retrieval_metrics.get('top3_accuracy', 0)),
                'oracle_accuracy_top5': oracle_retrieval_metrics.get('oracle_accuracy_top5', oracle_retrieval_metrics.get('top5_accuracy', 0)),
                'note': 'Upper bound - best achievable performance from candidate set'
            }
        
        if baseline_metrics:
            results_data['baseline'] = {
                'method': 'LTR (initial weights) on 5 candidates',
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
                    'current_to_oracle_top5': oracle_retrieval_metrics['oracle_accuracy_top5'] - metrics['top1_accuracy'],
                    'note': 'Room for improvement to reach oracle performance'
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed test results saved to: {filepath}")


def _find_data_file(filename: str) -> str:
    """
    Smart path resolution to find data files
    Tries multiple locations relative to script and working directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locations to try (in order)
    search_paths = [
        # Relative to current working directory
        filename,
        # Relative to script directory (parent)
        os.path.join(script_dir, '..', filename),
        # Relative to script directory (same level)
        os.path.join(script_dir, filename),
        # Absolute path if provided
        os.path.abspath(filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # If not found, return the first attempt (will raise error with clear message)
    return search_paths[0]


def main():
    """Main function"""
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_parent_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Linear Learning-to-Rank Model for Tool Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (auto-detect tool_pool.json and tool_calling_graphs.json)
  python learning_to_rank.py
  
  # Specify custom input files and output directory
  python learning_to_rank.py --tool_pool /path/to/tool_pool.json \\
                             --calling_graph /path/to/tool_calling_graphs.json \\
                             --output_dir ./results
  
  # Adjust training parameters
  python learning_to_rank.py --lr 0.05 --epochs 200 --lambda_reg 0.01
  
  # Use a different BERT model
  python learning_to_rank.py --model sentence-transformers/all-mpnet-base-v2
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
        help='Path to tool_calling_graphs.json (for compatibility, not used in dataset preparation)'
    )
    
    parser.add_argument(
        '--tool_selection',
        type=str,
        default=None,
        help='Path to tool_selection_results.json (default: auto-detect from output dir)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(default_parent_dir, 'output'),
        help='Output directory for results (default: ../output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='BERT model from Hugging Face (default: sentence-transformers/all-MiniLM-L6-v2)'
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
    
    # Auto-detect data files if not specified
    if args.tool_pool is None:
        args.tool_pool = _find_data_file('tool_pool.json')
    if args.calling_graph is None:
        args.calling_graph = _find_data_file('tool_calling_graphs.json')
    if args.tool_selection is None:
        # Try to find tool_selection_results.json in output directory
        tool_selection_path = os.path.join(args.output_dir, 'tool_selection_results.json')
        if not os.path.exists(tool_selection_path):
            # Fall back to auto-detect
            tool_selection_path = _find_data_file('tool_selection_results.json')
        args.tool_selection = tool_selection_path
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Linear Learning-to-Rank for Tool Selection")
    print("Using BERT Embeddings (transformers library)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Tool Pool:         {args.tool_pool}")
    print(f"  Calling Graph:     {args.calling_graph}")
    print(f"  Tool Selection:    {args.tool_selection}")
    print(f"  Output Dir:        {args.output_dir}")
    print(f"  Model:             {args.model}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  L2 Reg:            {args.lambda_reg}")
    print(f"  Max Traces:        {args.max_traces if args.max_traces else 'All'}")
    print("="*80 + "\n")
    
    # Initialize model
    ltr = LinearLearningToRank(args.tool_pool, args.calling_graph, model_name=args.model)
    
    # Prepare dataset from tool_selection results
    train_data, test_data = ltr.prepare_dataset(
        calling_graph_path=args.calling_graph,
        tool_selection_path=args.tool_selection,
        max_traces=args.max_traces
    )
    
    if len(train_data) == 0:
        print("\nError: No training data available!")
        return
    
    # Evaluate direct retrieval baseline (no candidate selection)
    direct_retrieval_metrics = ltr.evaluate_direct_retrieval(tool_selection_path=args.tool_selection)
    
    # Evaluate oracle retrieval baseline (best possible from candidates)
    # Pass test_data to filter to test set only (for fair comparison)
    oracle_retrieval_metrics = ltr.evaluate_oracle_retrieval(
        test_data=test_data, 
        tool_selection_path=args.tool_selection
    )
    
    # Evaluate LTR baseline (before training, on 5 candidates)
    print("\n" + "="*80)
    print("LTR BASELINE PERFORMANCE (Before Training, on 5 candidates)")
    print("="*80)
    print("Testing with initial weights (all 1.0) on retrieved 5 candidates...")
    baseline_metrics = ltr.evaluate(test_data, verbose=False)
    print(f"\nLTR Baseline Results:")
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
    print("TRAINED MODEL PERFORMANCE (After Training)")
    print("="*80)
    metrics = ltr.evaluate(test_data, verbose=True)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\n{'Method':<55} {'Top-1 Acc':<15}")
    print(f"{'-'*70}")
    print(f"{'1. Direct Retrieval (no candidate selection)':<55} {direct_retrieval_metrics['top1_accuracy']:.2%}")
    print(f"{'2. Oracle Retrieval (best from top-5 candidates)':<55} {oracle_retrieval_metrics['oracle_accuracy_top5']:.2%}  [Upper Bound]")
    print(f"{'3. LTR Baseline (initial weights, 5 candidates)':<55} {baseline_metrics['top1_accuracy']:.2%}")
    print(f"{'4. LTR Trained (trained weights, 5 candidates)':<55} {metrics['top1_accuracy']:.2%}")
    print(f"\n{'Improvement:':<55}")
    print(f"{'  vs Direct Retrieval:':<55} {'+' if metrics['top1_accuracy'] > direct_retrieval_metrics['top1_accuracy'] else ''}{(metrics['top1_accuracy']-direct_retrieval_metrics['top1_accuracy'])*100:.2f}%")
    print(f"{'  vs LTR Baseline:':<55} {'+' if metrics['top1_accuracy'] > baseline_metrics['top1_accuracy'] else ''}{(metrics['top1_accuracy']-baseline_metrics['top1_accuracy'])*100:.2f}%")
    print(f"{'  Gap to Oracle (room for improvement):':<55} {(oracle_retrieval_metrics['oracle_accuracy_top5']-metrics['top1_accuracy'])*100:.2f}%")
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
