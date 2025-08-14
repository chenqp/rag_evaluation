
import requests
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
from config import settings
from logger_config import logger
import llm_evaluator
from base_rag_evalutor import BaseRagEvaluator
'''
对dif的知识库检索和rag生成质量进行评估
'''
class DifyRagEvaluator(BaseRagEvaluator):

    def get_metrics(self) -> list:
        """定义dify RAG评估指标"""
        evaluator_llm=llm_evaluator.get_evaluator_llm()
        
        # 定义评估指标
        self.metrics = [
            LLMContextPrecisionWithReference(llm=evaluator_llm),     #ContextPrecision,上下文精度，衡量 retrieved_contexts 中相关块比例的指标，计算方法是上下文中的每个块的 precision@k 的平均值
            LLMContextRecall(llm=evaluator_llm),                     #ContextRecall,上下文召回率：衡量成功检索到的相关文档（或信息片段）的比率
            ContextRelevance(llm=evaluator_llm),                     #ContextRelevance,上下文相关性：评估检索到的上下文（块或段落）是否与用户输入相关
            Faithfulness(llm=evaluator_llm),                         #Faithfulness,忠实度:衡量RAG响应与检索到的上下文的事实一致性
            AnswerAccuracy(llm=evaluator_llm),                       #AnswerAccuracy,回答准确性：衡量RAG响应与给定问题的参考标准答案之间的一致性
        ]
        return self.metrics,evaluator_llm
        
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
        
            headers = {'Authorization': f'Bearer {api_key}'}

            if not dataset_id:
                logger.error("数据集ID未提供")
                raise ValueError("数据集ID未提供")
            
            search_method=retrieve_config.get("search_method")
            top_k=retrieve_config.get("top_k")
            reranking_enable=retrieve_config.get("reranking_enable")
            reranking_mode=retrieve_config.get("reranking_mode",None)  # weighted_score,reranking_mode
            reranking_model_name=retrieve_config.get("reranking_model_name")
            reranking_provider_name =retrieve_config.get("reranking_provider_name","langgenius/xinference/xinference")

            score_threshold_enabled=retrieve_config.get("score_threshold_enabled")
            score_threshold=retrieve_config.get("score_threshold")
           
            embedding_model_name=retrieve_config.get("embedding_model_name","bge-m3")
            embedding_provider_name =retrieve_config.get("embedding_provider_name","langgenius/xinference/xinference")
            
            vector_weight=retrieve_config.get("vector_weight",0.9)
            keyword_weight=retrieve_config.get("keyword_weight",0.1)

            if reranking_mode == "weighted_score":
                weights = {
                    "keyword_setting": {"keyword_weight":keyword_weight},
                    "vector_setting": {
                        "embedding_model_name": embedding_model_name,
                        "embedding_provider_name": embedding_provider_name,
                        "vector_weight": vector_weight,
                        
                    },
                    "weight_type": "customized"
                }
            else:
                weights = None
           
            # 构建请求体
            payload = {
                "query": query,
                "retrieval_model": {
                    "search_method": search_method ,
                    "reranking_enable": reranking_enable ,
                    "reranking_mode": reranking_mode,  
                    "reranking_model": {
                        "reranking_provider_name": reranking_provider_name,
                        "reranking_model_name": reranking_model_name
                    },
                    "weights": weights,
                    "top_k": top_k ,
                    "score_threshold_enabled": score_threshold_enabled ,
                    "score_threshold": score_threshold ,
                    "metadata_filtering_conditions":  {
                        "logical_operator": "and",
                        "conditions": []
                    }
                }
            }
            
             # 显示详细的配置参数
            logger.info(f"开始检索知识库: {dataset_id}")
            logger.info(f"检索参数 - 查询问题: {query}, 搜索方法: {search_method}, 启用重排序: {reranking_enable}")
                       
            if reranking_enable:
                logger.info(f"检索配置 rerank model : {reranking_model_name}")
            logger.info(f"检索配置 - top_k: {top_k}")
            logger.info(f"检索配置 - 分数阈值启用: {score_threshold_enabled}")
            if score_threshold_enabled :
                logger.info(f"检索配置 - 分数阈值: {score_threshold}")

            if weights:
                logger.info(f"混合检索 - 权重设置: {weights}")
            
            response = requests.post(url, headers=headers, json=payload)
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
            rag_response = self.get_rag_response(query,dify_client)
            rag_response = BaseRagEvaluator.clean_rag_answer(rag_response)
            
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
    def get_rag_response(self,question: str,dify_client:OpenAI) -> str:
        """
        通过OpenAI兼容接口获取RAG工作流的response
        
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
            
            return answer
        except Exception as e:
            logger.error(f"获取RAG答案失败: {e}")
            return ""

    def run_evaluation(self):
        """
        评估Dify知识库召回内容和rag工作流的回复
        """
        logger.info("开始评估Dify知识库检索和RAG生成的质量...")

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
            
            for retrieve_config in rag_config["retrieve_configs"]:
                retrieve_name=retrieve_config.get("name")
                # 创建RAGAS数据集
                logger.info(f"开始创建RAGAS数据集,检索策略:{retrieve_name}...")
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
                    
            self.save_results(evaluation_results,result_file)
def main():

    try:
        # 执行评估
        dify_rag_evaluator = DifyRagEvaluator()
        dify_rag_evaluator.run_evaluation()
       
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()
