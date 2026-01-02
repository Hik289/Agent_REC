# 项目信息

## 项目名称
**Agent System Recommender** - 基于学习排序的智能体系统推荐框架

## 项目概述
本项目实现了两个推荐系统:
1. **单智能体推荐器** (Single Agent Recommender) - 用于单智能体任务的工具选择
2. **多智能体系统推荐器** (Multi-Agent System Recommender) - 用于多智能体任务的基于图的智能体系统选择

两个系统都采用两阶段方法:
- **阶段1**: 使用BERT嵌入相似度进行候选检索
- **阶段2**: 使用学习排序(LTR)模型进行最终选择

## 技术栈
- **Python**: 3.8+
- **深度学习框架**: PyTorch
- **NLP模型**: Transformers (Hugging Face)
- **机器学习**: scikit-learn
- **数据可视化**: matplotlib
- **数值计算**: numpy, scipy

## 项目结构

```
agent-system-recommender/
├── README.md                           # 项目说明文档
├── LICENSE                             # MIT开源协议
├── requirements.txt                    # Python依赖包
├── .gitignore                          # Git忽略文件配置
├── PROJECT_INFO.md                     # 项目信息(本文件)
│
├── data/                               # 数据集目录
│   ├── agent-data_protocol/
│   ├── Agents_Failure_Attribution/
│   ├── GTA/
│   ├── GUI-360/
│   ├── MCPToolBenchPP/
│   ├── MedAgentBench/
│   ├── Seal-Tools/
│   └── trail-benchmark/
│       ├── tool_pool.json              # 工具池定义
│       └── tool_calling_graphs.json    # 工具调用图
│
├── single agent recommender/           # 单智能体推荐器
│   ├── tool_selection.py              # 第1步: 基于嵌入的工具检索
│   ├── learning_to_rank.py            # 第2步: LTR模型训练
│   └── visualize_results.py           # 第3步: 结果可视化
│
├── multi-agent system recommender/     # 多智能体系统推荐器
│   ├── generate_node_candidates.py    # 第1步: 生成候选系统
│   ├── graph_retrieval.py             # 第2步: 图检索
│   ├── learning_to_rank.py            # 第3步: LTR模型训练
│   └── visualize_results.py           # 第4步: 结果可视化
│
├── output/                             # 输出目录
│   ├── tool_selection_results.json    # 工具选择结果
│   ├── ltr_model_weights.json         # 单智能体LTR模型权重
│   ├── ltr_test_results.json          # 单智能体测试结果
│   ├── node_candidates.json           # 节点候选系统
│   ├── graph_selection_results.json   # 图检索结果
│   ├── graph_ltr_model_weights.json   # 多智能体LTR模型权重
│   └── graph_ltr_test_results.json    # 多智能体测试结果
│
└── figure/                             # 可视化图表
    ├── single_agent_weights.png
    ├── single_agent_test_performance.png
    ├── single_agent_ground_truth_retrieval.png
    ├── single_agent_tool_selection_stats.png
    ├── multi_agent_system_weights.png
    ├── multi_agent_system_test_performance.png
    └── multi_agent_system_graph_selection_stats.png
```

## 核心功能

### 单智能体推荐器
- **工具选择**: 使用Sentence-BERT进行语义相似度计算
- **学习排序**: 4个特征函数
  - φ_rel: 工具-查询相关性
  - φ_hist: 历史可靠性
  - φ_coop: 图感知兼容性
  - φ_struct: 结构实用性

### 多智能体系统推荐器
- **候选生成**: 为每个节点生成随机调用树
- **图检索**: 使用BERT嵌入进行图语义匹配
- **学习排序**: 4个特征函数
  - φ_rel: 语义对齐(查询-系统)
  - φ_hist: 系统可靠性
  - φ_coop: 内部协作
  - φ_struct: 结构实用性

## 评估指标
- **Top-1准确率**: 正确工具/系统排名第一的查询百分比
- **Top-3准确率**: 正确工具/系统在前3名的查询百分比
- **MRR (平均倒数排名)**: 正确答案倒数排名的平均值

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行单智能体推荐器
```bash
cd "single agent recommender"
python tool_selection.py --tool_pool ../data/your_dataset/tool_pool.json
python learning_to_rank.py
python visualize_results.py
```

### 3. 运行多智能体系统推荐器
```bash
cd "multi-agent system recommender"
python generate_node_candidates.py --n_random 10
python graph_retrieval.py
python learning_to_rank.py
python visualize_results.py
```

## 代码特点
✅ 所有代码均已删除注释，保持简洁
✅ 完整的README文档说明
✅ MIT开源协议
✅ 完善的.gitignore配置
✅ 标准的Python项目结构
✅ 易于扩展和维护

## 适用场景
- 智能体工具推荐
- 多智能体系统选择
- 工具调用图分析
- 语义相似度计算
- 学习排序应用

## 上传到GitHub步骤

### 初始化Git仓库
```bash
cd /Users/manifect/Desktop/tt
git init
git add .
git commit -m "Initial commit: Agent System Recommender"
```

### 连接到GitHub远程仓库
```bash
git remote add origin https://github.com/yourusername/agent-system-recommender.git
git branch -M main
git push -u origin main
```

## 开发者信息
- **许可证**: MIT License
- **Python版本**: 3.8+
- **维护状态**: 活跃

## 贡献指南
欢迎提交Pull Request和Issue!

## 致谢
- Hugging Face Transformers团队
- Sentence-BERT项目
- 开源社区的支持

---
**准备就绪，可以上传到GitHub了! 🚀**

