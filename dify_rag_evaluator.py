import asyncio
import pandas as pd
import requests
import json
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerAccuracy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextRelevance
)

from ragas import SingleTurnSample, EvaluationDataset
from openai import OpenAI
import re
from config import settings
from logger_config import logger
import llm_evaluator

'''
对dif的知识库检索和rag工作流生成质量进行评估
'''
class DifyRagEvaluator:
    def __init__(self):
        asyncio.run(self.set_adapt_prompts())
    async def set_adapt_prompts(self):
        evaluator_llm=llm_evaluator.get_evaluator_llm()
        logger.info("正在修改评估prompt适配中文......")

        llm_context_precision =LLMContextPrecisionWithReference(llm=evaluator_llm)
        await llm_context_precision.adapt_prompts(language='chinese',llm=evaluator_llm)

        llm_context_recall=LLMContextRecall(llm=evaluator_llm)
        await llm_context_recall.adapt_prompts(language='chinese',llm=evaluator_llm)

        context_relevance=ContextRelevance(llm=evaluator_llm)
        await context_relevance.adapt_prompts(language='chinese',llm=evaluator_llm)

        faithfulness=Faithfulness(llm=evaluator_llm)
        await faithfulness.adapt_prompts(language='chinese',llm=evaluator_llm)

        answer_accuracy=AnswerAccuracy(llm=evaluator_llm)
        await answer_accuracy.adapt_prompts(language='chinese',llm=evaluator_llm)
        logger.info("评估prompt语言已适配为中文")

        # 定义评估指标
        self.metrics = [
            llm_context_precision,                  #ContextPrecision,上下文精度，衡量 retrieved_contexts 中相关块比例的指标，计算方法是上下文中的每个块的 precision@k 的平均值
            llm_context_recall,                     #ContextRecall,上下文召回率：衡量成功检索到的相关文档（或信息片段）的比率
            context_relevance,                      #ContextRelevance,上下文相关性：评估检索到的上下文（块或段落）是否与用户输入相关
            faithfulness,                           #Faithfulness,忠实度:衡量RAG响应与检索到的上下文的事实一致性
            answer_accuracy,                        #AnswerAccuracy,回答准确性：衡量RAG响应与给定问题的参考标准答案之间的一致性
        ]

    def prepare_evaluation_data(self,excel_file: str) -> List[Dict[str, Any]]:
        """
        从Excel文件中读取测试数据
        
        Args:
            excel_file: Excel文件路径
            
        Returns:
            包含问题和标准答案的列表
        """
        df = pd.read_excel(excel_file)
        evaluation_data = []
        
        # 从Excel文件中读取"问题"和"答案"列
        for _, row in df.iterrows():
            evaluation_data.append({
                "question": row.get("问题", ""),
                "ground_truth": row.get("答案", "")
            })
        
        return evaluation_data

    def retrieve(self,query: str,dify_rag_config:Dict[str, Any],
                            retrieve_config: Dict[str, Any]) -> List[str]:
        """
        使用检索Dify知识库内容
        
        Args:
            question: 检索问题
            dify_rag_config: Dify rag配置参数
            retrieve_config: 检索配置参数
            
        Returns:
            检索结果
        """

        try:
            base_url=dify_rag_config.get("knowledge_base_url")
            api_key=dify_rag_config.get("knowledge_api_key")
            dataset_id = dify_rag_config.get("dataset_id")

            url = f"{base_url}/v1/datasets/{dataset_id}/retrieve"
        
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            if not dataset_id:
                logger.error("数据集ID未提供")
                raise ValueError("数据集ID未提供")
            
            search_method=retrieve_config.get("search_method")
            top_k=retrieve_config.get("top_k")
            reranking_enable=retrieve_config.get("reranking_enable")
            reranking_model_name=retrieve_config.get("reranking_model_name")
            score_threshold_enabled=retrieve_config.get("score_threshold_enabled")
            score_threshold=retrieve_config.get("score_threshold")

            logger.info(f"开始检索知识库: {dataset_id}")
            logger.info(f"检索参数 - 查询问题: {query}, 搜索方法: {search_method}, 启用重排序: {reranking_enable}")
    
            # 构建请求体
            payload = {
                "query": query,
                "retrieval_model": {
                    "search_method": search_method or 'hybrid_search',
                    "reranking_enable": reranking_enable if reranking_enable is not None else False,
                    "reranking_mode": None,
                    "reranking_model": {
                        "reranking_provider_name": "",
                        "reranking_model_name": reranking_model_name
                    },
                    "weights": None,
                    "top_k": top_k ,
                    "score_threshold_enabled": score_threshold_enabled if score_threshold_enabled is not None else False,
                    "score_threshold": score_threshold if score_threshold is not None else None,
                    "metadata_filtering_conditions":  {
                        "logical_operator": "and",
                        "conditions": []
                    }
                }
            }
            
            # 显示详细的配置参数
            if reranking_enable:
                logger.info(f"检索配置 rerank model : {reranking_model_name}")
            logger.info(f"检索配置 - top_k: {payload['retrieval_model']['top_k']}")
            logger.info(f"检索配置 - 分数阈值启用: {payload['retrieval_model']['score_threshold_enabled']}")
            if payload['retrieval_model']['score_threshold_enabled'] :
                logger.info(f"检索配置 - 分数阈值: {payload['retrieval_model']['score_threshold']}")
            
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result=response.json()
            records = result.get("records", [])
            contexts = []

            for record in records[:top_k]:
                # 提取segment.content作为上下文
                segment = record.get("segment", {})
                content = segment.get("content", "")
                if content :
                    contexts.append(content)

            return contexts

        except Exception as e:
            logger.error(f"Dify检索失败: {e}")
            return []

    def create_evaluation_dataset(self,qa_data: List[Dict[str, Any]], rag_config:Dict[str, Any],retrieve_config: Dict[str, Any],
                                dify_client:OpenAI ) -> EvaluationDataset:
        """
        创建用于RAGAS评估的数据集
        
        Args:
            qa_data: QA测试数据列表
            rag_config: rag配置参数
            
        Returns:
            RAGAS格式的数据集
        """
        samples=[]

        for i, data in enumerate(qa_data):
            query = data["question"]
            ground_truth = data["ground_truth"]
            
            logger.info(f"处理QA对 {i+1}/{len(qa_data)}: {query}")
            
            # 使用Dify检索,获取检索到的上下文片段
            ctxs= self.retrieve( query,rag_config, retrieve_config)
            logger.info(f"检索到 {len(ctxs)} 个上下文片段")
            # 通过RAG工作流获取答案
            rag_response = self.get_rag_answer(query,dify_client)
            
            sample= SingleTurnSample(
                user_input=query,
                retrieved_contexts=ctxs,
                response=rag_response,
                reference=str(ground_truth)
            )
            samples.append(sample)
            logger.info(f"RAG系统生成的答案: {rag_response[:200]}..." if rag_response else "无答案")
        
        # 创建评估数据集
        evaluation_dataset = EvaluationDataset(samples)
        
        return evaluation_dataset

    def get_rag_answer(self,question: str,dify_client:OpenAI) -> str:
        """
        通过OpenAI兼容接口获取RAG工作流的answer
        
        Args:
            question: 用户提问
            dify_client: OpenAI兼容客户端
            
        Returns:
            RAG系统生成的答案
        """
        try:
            # 使用OpenAI客户端调用API
            response = dify_client.chat.completions.create(
                model="dify",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            
            # 提取答案
            answer = response.choices[0].message.content
            
            # 去除answer中的多余标签
            answer = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', answer)  # Handle multiline with [\s\S]
            answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
            answer = re.sub(r'think:\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'<.*?>', '', answer).strip()
            
            return answer
        except Exception as e:
            logger.error(f"获取RAG答案失败: {e}")
            return ""

    def evaluate_dify_rag(self):
        """
        评估Dify知识库召回内容和rag工作流的回复
        """
        logger.info("开始评估Dify知识库检索和RAG工作流生成的质量...")

        dify_rags=settings["dify_rags"]

        for rag_config in dify_rags:
            # 准备问答数据
            qa_excel_file=rag_config.get("qa_excel_file")
            qa_data = self.prepare_evaluation_data(qa_excel_file)
            logger.info(f"加载了 {len(qa_data)} 条测试数据")
            
            # 初始化dify OpenAI客户端
            dify_client = OpenAI(
                base_url=rag_config.get("rag_base_url"),
                api_key=rag_config.get("rag_api_key"),
            )

            result_file=rag_config.get("result_excel_file")
            evaluation_results = []

            
            for retrieve_config in settings["retrieve_configs"]:
                retrieve_name=retrieve_config.get("name")
                # 创建RAGAS数据集
                evaluation_dataset = self.create_evaluation_dataset(qa_data,
                                                                rag_config=rag_config,
                                                                retrieve_config=retrieve_config,
                                                                dify_client=dify_client)
                # 执行评估
                logger.info(f"开始执行RAGAS评估,检索策略:{retrieve_name}...")
                result = evaluate(
                    dataset=evaluation_dataset,
                    metrics=self.metrics,
                
                )
            
                logger.info(f"检索策略:{retrieve_name}评估完成")
                
                # 保存结果
                result_df = result.to_pandas()
                evaluation_results.append({'retrieve_name': retrieve_name, 'result_df': result_df})
                    
            with pd.ExcelWriter(result_file,engine='xlsxwriter') as writer:
                for result in evaluation_results:
                    result_df = result['result_df']
                    retrieve_name = result['retrieve_name']
                    result_df.to_excel(writer, index=False,sheet_name=retrieve_name)
            logger.info(f"评估结果已保存到{result_file}")
def main():

    try:
        # 执行评估
        dify_rag_evaluator = DifyRagEvaluator()
        dify_rag_evaluator.evaluate_dify_rag()
       
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()
