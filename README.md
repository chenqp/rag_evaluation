# Dify 知识库检索和RAG工作流生成质量自动化评估工具

进行Dify知识库检索和RAG工作流生成质量的自动化评估，支持多组知识库和RAG工作流评估，支持多种检索方法和配置选项。

## 评估指标

- ContextPrecision：上下文精度，衡量 retrieved_contexts 中相关块比例的指标，计算方法是上下文中的每个块的 precision@k 的平均值
- ContextRecall：上下文召回率，从知识库中检索到的相关文档数量相较于总相关文档数量的比值
- ContextRelevance：上下文相关性，评估检索到的上下文（块或段落）是否与用户输入相关
- Faithfulness：忠实度，衡量RAG响应与检索到的上下文的事实一致性
- AnswerAccuracy：回答准确性，衡量RAG响应与给定问题的参考标准答案之间的一致性

## 功能特点

- 支持多组知识库和RAG工作流评估
- 支持多种检索方法：关键词搜索、语义搜索、全文搜索、混合搜索
- 支持设置重排序功能
- 可配置检索top-k返回数量
- 支持 Score 阈值过滤
- 支持通过配置文件管理多组参数评估配置
- 使用RAGAS框架评估检索和生成效果
- 支持使用自定义LLM（如DeepSeek）进行评估

## 用uv自动创建虚拟环境，安装依赖
不想科学上网，可以安装uv国内加速版 https://gitee.com/wangnov/uv-custom/releases
项目文件夹下运行：
```
uv sync
```

## 配置

### 配置文件

配置文件 `setting.toml` 使用 TOML 格式，支持定义多组测试配置。

配置示例：

```toml

#支持多组知识库和rag评估
#difyde的知识库和RAG工作流配置，需安装插件OpenAI Compatible Dify App,并启用api端点将集成了知识库的工作流转换为openai兼容的api
[[dify_rag]]

#open ai api兼容的评估模型
[evaluator_llm]

# 使用数组形式定义的多组测试配置
[[retrieve_configs]]
name = "hybrid_search1"
search_method = "hybrid_search"
top_k = 3
reranking_enable = false
reranking_model_name = "bge-reranker-v2-m3"
score_threshold_enabled = false
score_threshold = 0.5

[[retrieve_configs]]
name = "hybrid_search2"
search_method = "hybrid_search"
top_k = 3
reranking_enable = true
reranking_model_name = "bge-reranker-v2-m3"
score_threshold_enabled = true
score_threshold = 0.6

配置参数说明：

- name: 配置名称，用于标识不同的测试配置,会作为写入测试结果excel的sheet名称
- search_method: 检索方法：keyword_search, semantic_search, full_text_search, hybrid_search
- reranking_enable: 是否启用重排序功能
- reranking_model_name: 重排序模型名称
- top_k: 返回结果数量
- score_threshold_enabled: 是否启用 Score 阈值过滤
- score_threshold: Score 阈值

```

### 使用

使用RAGAS框架评估知识库检索和rag工作流生成质量。评估脚本会自动读取Excel文件中的测试问题和标准答案，并使用指定的LLM进行评估。要使用评估功能，请按照以下流程：

1. 准备好包含测试问题和标准答案的Excel文件,应至少包含'问题'和'答案'2个列（第一个知识库默认qa1.xlsx，可在settings.toml中修改）
2. 设置dify知识库和rag工作流(在settings.toml修改)
3. 在settings.toml配置评估LLM相关参数,api_key在.secrets.toml设置
4. 在项目文件夹运行评估脚本：

```bash
uv run main.py
或者
uv run dify_rag_evaluator.py
```

评估结果默认保存在 evaluation_results1.xlsx,可在settings.toml中修改。
metrics_plot.py根据结果文件，绘制指标的直方图分布

