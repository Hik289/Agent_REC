# Learning to Rank 评估指标算法详解

本文档详细说明了Multi-Agent System和Single Agent推荐器中Learning to Rank模型的评估指标计算方法。

---

## 目录
1. [核心评估指标](#核心评估指标)
2. [Single Agent LTR指标计算](#single-agent-ltr指标计算)
3. [Multi-Agent System LTR指标计算](#multi-agent-system-ltr指标计算)
4. [特征函数详解](#特征函数详解)
5. [损失函数](#损失函数)

---

## 核心评估指标

### 1. Top-1 Accuracy (Top-1准确率)
**定义**: 模型预测的Top-1结果与真实答案匹配的查询比例

**计算公式**:
```
Top-1 Accuracy = (正确预测在Rank 1的查询数) / (总查询数)
```

**代码实现**:
```python
if rank == 1:
    top1_correct += 1

top1_accuracy = top1_correct / total_queries
```

**含义**: 衡量模型将正确答案排在第一位的能力

---

### 2. Top-3 Accuracy (Top-3准确率)
**定义**: 模型预测的Top-3结果中包含真实答案的查询比例

**计算公式**:
```
Top-3 Accuracy = (正确答案在Top-3内的查询数) / (总查询数)
```

**代码实现**:
```python
if rank <= 3:
    top3_correct += 1

top3_accuracy = top3_correct / total_queries
```

**含义**: 衡量模型在前3个候选中找到正确答案的能力

---

### 3. MRR (Mean Reciprocal Rank - 平均倒数排名)
**定义**: 所有查询中正确答案排名的倒数的平均值

**计算公式**:
```
MRR = (1/N) × Σ(1/rank_i)
```
其中:
- N = 总查询数
- rank_i = 第i个查询中正确答案的排名

**代码实现**:
```python
mrr_sum += 1.0 / rank

mrr = mrr_sum / total_queries
```

**示例**:
- 如果正确答案排名第1: 贡献 1/1 = 1.0
- 如果正确答案排名第2: 贡献 1/2 = 0.5
- 如果正确答案排名第3: 贡献 1/3 = 0.333
- 如果正确答案排名第4: 贡献 1/4 = 0.25

**含义**: 综合考虑排名质量，排名越靠前贡献越大

---

## Single Agent LTR指标计算

### 评估流程

#### 1. 数据准备
```python
# 按查询分组
query_groups = {}
for sample in test_data:
    key = (sample['trace_id'], sample['node_id'])
    if key not in query_groups:
        query_groups[key] = []
    query_groups[key].append(sample)
```

#### 2. 为每个查询计算排名
```python
for group_samples in query_groups.values():
    # 获取特征矩阵和标签
    features_matrix = np.array([s['features'] for s in group_samples])
    labels = np.array([s['label'] for s in group_samples])
    tools = [s['tool'] for s in group_samples]
    
    # 计算分数: s = w^T × Φ
    scores = features_matrix @ self.w
    
    # 排序得到排名索引
    ranked_indices = np.argsort(scores)[::-1]  # 降序排列
    
    # 找到正确答案的排名
    true_idx = np.where(labels == 1)[0][0]
    rank = np.where(ranked_indices == true_idx)[0][0] + 1
```

#### 3. 更新指标
```python
total_queries += 1
is_top1 = (rank == 1)
is_top3 = (rank <= 3)

if is_top1:
    top1_correct += 1
if is_top3:
    top3_correct += 1
mrr_sum += 1.0 / rank
```

#### 4. 计算最终指标
```python
metrics = {
    'top1_accuracy': top1_correct / total_queries,
    'top3_accuracy': top3_correct / total_queries,
    'mrr': mrr_sum / total_queries,
    'num_queries': total_queries
}
```

### Single Agent特征函数

**评分公式**: `s(q, a) = w^T × Φ(q, a)`

其中 `Φ(q, a) = [φ_rel, φ_hist, φ_coop, φ_struct]`

#### φ_rel: 工具-查询相关性
```python
def phi_rel(query: str, tool_name: str) -> float:
    query_embedding = self._get_embedding(query)
    tool_embedding = self.tool_embeddings[tool_idx]
    
    similarity = cosine_similarity(query_embedding, tool_embedding)
    return (similarity + 1) / 2  # 归一化到[0, 1]
```

#### φ_hist: 历史可靠性
```python
def phi_hist(tool_name: str) -> float:
    return self.tool_reliability.get(tool_name, 0.5)
```

#### φ_coop: 图感知兼容性
```python
def phi_coop(query: str, tool_name: str) -> float:
    if tool_name.lower() in query.lower():
        return 1.0
    elif any(word in query.lower() for word in tool_name.split('_')):
        return 0.7
    else:
        return 0.3
```

#### φ_struct: 结构实用性
```python
def phi_struct(query: str, tool_name: str) -> float:
    phi_rel_val = self.phi_rel(query, tool_name)
    return sqrt(phi_rel_val + 0.1)
```

---

## Multi-Agent System LTR指标计算

### 评估流程

#### 1. 数据准备（与Single Agent相同）
```python
query_groups = {}
for sample in test_data:
    key = (sample['trace_id'], sample['node_id'])
    if key not in query_groups:
        query_groups[key] = []
    query_groups[key].append(sample)
```

#### 2. 为每个查询计算排名
```python
for group_samples in query_groups.values():
    features_matrix = np.array([s['features'] for s in group_samples])
    labels = np.array([s['label'] for s in group_samples])
    candidate_ids = [s['candidate_id'] for s in group_samples]
    
    # 计算分数: s = w^T × Φ
    scores = features_matrix @ self.w
    
    # 排序
    ranked_indices = np.argsort(scores)[::-1]
    
    # 找到正确答案排名
    true_idx = np.where(labels == 1)[0][0]
    rank = np.where(ranked_indices == true_idx)[0][0] + 1
```

#### 3. 更新指标（与Single Agent相同）
```python
total_queries += 1
is_top1 = (rank == 1)
is_top3 = (rank <= 3)

if is_top1:
    top1_correct += 1
if is_top3:
    top3_correct += 1
mrr_sum += 1.0 / rank
```

#### 4. 详细结果记录
```python
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
```

### Multi-Agent特征函数

**评分公式**: `s(q, g) = w^T × Φ(q, g)`

其中 `Φ(q, g) = [φ_rel, φ_hist, φ_coop, φ_struct]`

#### φ_rel: 语义对齐（查询-系统）
```python
def phi_rel(query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
    tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
    tree_text = self._tree_to_text(tree)
    
    query_embedding = self._get_embedding(query)
    tree_embedding = self._get_embedding(tree_text)
    
    similarity = cosine_similarity(query_embedding, tree_embedding)
    return (similarity + 1) / 2  # 归一化到[0, 1]
```

#### φ_hist: 系统可靠性
```python
def phi_hist(candidate_type: str) -> float:
    # 基于候选类型的可靠性分数
    return self.graph_reliability.get(candidate_type, 0.5)
```

#### φ_coop: 内部协作
```python
def phi_coop(query: str, candidate_id: str, trace_id: str, node_id: str) -> float:
    tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
    num_nodes = len(tree.get('nodes', []))
    query_complexity = len(query.split())
    
    if 'ground_truth' in candidate_id:
        return 0.9
    
    if query_complexity < 5:
        size_score = 1.0 / (1.0 + log1p(num_nodes))
    else:
        size_score = min(1.0, log1p(num_nodes) / 5.0)
    
    return size_score * 0.8 + 0.2
```

#### φ_struct: 结构实用性
```python
def phi_struct(candidate_id: str, trace_id: str, node_id: str, 
               all_candidate_ids: List[str]) -> float:
    tree = self._get_candidate_tree(trace_id, node_id, candidate_id)
    num_nodes = len(tree.get('nodes', []))
    num_edges = len(tree.get('edges', []))
    
    # 平衡分数
    depth = num_edges / max(num_nodes, 1) if num_nodes > 0 else 0
    balance_score = min(1.0, depth)
    
    # 唯一性分数
    sizes = [len(self._get_candidate_tree(trace_id, node_id, cid).get('nodes', []))
             for cid in all_candidate_ids if cid != candidate_id]
    
    if sizes:
        avg_size = mean(sizes)
        uniqueness = abs(num_nodes - avg_size) / max(avg_size, 1)
        uniqueness_score = min(1.0, uniqueness / 2.0)
    else:
        uniqueness_score = 0.5
    
    return balance_score * 0.6 + uniqueness_score * 0.4
```

---

## 损失函数

### Softmax Cross-Entropy Loss with L2 Regularization

**公式**:
```
L = -Σ(y_i × log(p_i)) + λ × ||w||²
```

其中:
- `y_i`: 真实标签（0或1）
- `p_i`: Softmax概率
- `λ`: L2正则化系数
- `w`: 权重向量

**代码实现**:
```python
# 计算分数
scores = features_matrix @ self.w

# Softmax概率
probs = softmax(scores)

# 交叉熵损失
loss = -np.sum(labels * np.log(probs + 1e-10))

# L2正则化
loss += lambda_reg * np.sum(self.w ** 2)

# 梯度
grad = features_matrix.T @ (probs - labels) + 2 * lambda_reg * self.w

# 梯度下降
self.w -= learning_rate * grad
```

---

## 训练过程

### 1. 分组
```python
query_groups = {}
for sample in train_data:
    key = (sample['trace_id'], sample['node_id'])
    if key not in query_groups:
        query_groups[key] = []
    query_groups[key].append(sample)
```

### 2. 迭代训练
```python
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
        
        # 前向传播
        scores = features_matrix @ self.w
        probs = softmax(scores)
        
        # 计算损失
        loss = -np.sum(labels * np.log(probs + 1e-10))
        loss += lambda_reg * np.sum(self.w ** 2)
        total_loss += loss
        
        # 反向传播
        grad = features_matrix.T @ (probs - labels) + 2 * lambda_reg * self.w
        self.w -= learning_rate * grad
    
    avg_loss = total_loss / max(num_groups, 1)
```

---

## 指标对比表

| 指标 | 含义 | 值域 | 最佳值 | 说明 |
|------|------|------|--------|------|
| **Top-1 Accuracy** | 第一名命中率 | [0, 1] | 1.0 | 越高越好，1.0表示完美 |
| **Top-3 Accuracy** | 前三名命中率 | [0, 1] | 1.0 | 越高越好，通常高于Top-1 |
| **MRR** | 平均倒数排名 | [0, 1] | 1.0 | 越高越好，考虑排名质量 |

---

## 示例计算

### 场景1: 5个查询的评估

| 查询 | 正确答案排名 | Top-1? | Top-3? | MRR贡献 |
|------|-------------|--------|--------|---------|
| Q1   | 1           | ✓      | ✓      | 1.0     |
| Q2   | 2           | ✗      | ✓      | 0.5     |
| Q3   | 1           | ✓      | ✓      | 1.0     |
| Q4   | 3           | ✗      | ✓      | 0.333   |
| Q5   | 1           | ✓      | ✓      | 1.0     |

**计算结果**:
- Top-1 Accuracy = 3/5 = 0.6 = 60%
- Top-3 Accuracy = 5/5 = 1.0 = 100%
- MRR = (1.0 + 0.5 + 1.0 + 0.333 + 1.0) / 5 = 0.767

### 场景2: 性能较差的情况

| 查询 | 正确答案排名 | Top-1? | Top-3? | MRR贡献 |
|------|-------------|--------|--------|---------|
| Q1   | 2           | ✗      | ✓      | 0.5     |
| Q2   | 4           | ✗      | ✗      | 0.25    |
| Q3   | 3           | ✗      | ✓      | 0.333   |
| Q4   | 1           | ✓      | ✓      | 1.0     |
| Q5   | 3           | ✗      | ✓      | 0.333   |

**计算结果**:
- Top-1 Accuracy = 1/5 = 0.2 = 20%
- Top-3 Accuracy = 4/5 = 0.8 = 80%
- MRR = (0.5 + 0.25 + 0.333 + 1.0 + 0.333) / 5 = 0.483

---

## 关键区别总结

### Single Agent vs Multi-Agent

| 方面 | Single Agent | Multi-Agent System |
|------|--------------|-------------------|
| **候选对象** | 工具 (Tools) | 图/系统 (Graphs) |
| **特征空间** | 工具属性 | 图结构属性 |
| **φ_rel** | 工具-查询语义相似度 | 图-查询语义相似度 |
| **φ_hist** | 工具历史使用频率 | 系统历史可靠性 |
| **φ_coop** | 工具名称匹配 | 图复杂度匹配 |
| **φ_struct** | 相关性变换 | 图结构平衡性+唯一性 |
| **评估指标** | Top-1, Top-3, MRR (相同) | Top-1, Top-3, MRR (相同) |

---

## 使用建议

### 1. 指标选择
- **Top-1 Accuracy**: 关注最优解场景
- **Top-3 Accuracy**: 关注候选集质量
- **MRR**: 综合评估排序质量

### 2. 阈值设置
- Top-1 Accuracy > 0.5: 基本可用
- Top-1 Accuracy > 0.7: 良好性能
- Top-1 Accuracy > 0.9: 优秀性能

### 3. 调优方向
- 如果Top-1低但Top-3高: 调整权重分布
- 如果MRR低: 关注特征质量
- 如果所有指标都低: 检查特征设计和数据质量

---

**文档版本**: 1.0  
**最后更新**: 2025-01-02

