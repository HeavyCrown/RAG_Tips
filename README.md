# 一些调优的方法：
## 基于文本数据和分片策略：
### Chunk_size的选择（文档处理）
1. 构造出可选的chunk_size的列表[128，256，...]
2. 构造基于文档的questions
		'data_generator = DatasetGenerator.from_documents(eval_documents)
        eval_questions = data_generator.generate_questions_from_nodes()'
3. 使用大模型作为评估模型，记录每个chunk_size的指标（平均检索时间，平均忠诚度和平均相关性）
4. 选择指标最优的chunk_size
### 命题分片（文档处理）
1. 原则：将文章分解成简洁、完整、有意义的句子以便更好地控制和处理特定查询(特别是提取知识)。事实性语句，也叫命题（xxx is xxx）
2. 实现：LLM+Prompt形成文本内容的事实生成（few-shot prompting）
3. 评估：基于大模型的准确性、清晰性、简洁性和完整性评估
## Query增强：
### Query转换技术：可以单独使用，也可以结合使用
1. Rewriting：通过LLM将查询变得更加详细、具体
2. step-back prompting：通过LLM生成更加广泛、通用的query，帮助检索相关的背景信息
3. 子查询分解：通过大模型将复杂问题分解，以便更加全面的查询
### 生成假设性文档
1. 通过llm将query生成为一个字数为chunk_size的文档
2. 好处：
	1. 扩展成完整文档后，可以捕获更细微和相关的匹配
	2. 对于难以直接匹配的复杂查询特别有效
	3. 可以更好地捕获原始问题背后的上下文和意图
