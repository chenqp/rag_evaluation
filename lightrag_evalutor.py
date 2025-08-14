
import json
import llm_evaluator
from ragas.metrics import (
    Faithfulness,
    AnswerAccuracy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextRelevance
)
import requests
from typing import List, Dict, Any,Optional
from config import settings
from logger_config import logger
from ragas import SingleTurnSample, EvaluationDataset

from ragas import evaluate
import re
from base_rag_evalutor import BaseRagEvaluator

class LightRagEvaluator(BaseRagEvaluator):
    def get_metrics(self) -> list:
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
   
    def create_evaluation_dataset(self,qa_data: List[Dict[str, Any]], 
                                  rag_config:Dict[str, Any],
                                  retrieve_config: Dict[str, Any],
                                  ) -> EvaluationDataset:
        """
        创建用于RAGAS评估的数据集
        
        Args:
            qa_data: QA测试数据列表
            lightrag_config: lightrag配置参数
            
        Returns:
            RAGAS格式的数据集
        """
        samples=[]

        for i, data in enumerate(qa_data):
            query = data["question"]
            ground_truth = data["ground_truth"]
            
            logger.info(f"处理QA对 {i+1}/{len(qa_data)}: {query}")

            # 获取检索的上下文片段
            ctxs= self.retrieve_context( query,rag_config, retrieve_config)
            # 通过获取答案
            rag_response = self.retrieve(query,rag_config,retrieve_config,only_need_context=False)
            rag_response= BaseRagEvaluator.clean_rag_answer(rag_response)
            
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
    
    def retrieve(
        self,
        query: str,
        lightrag_config:Dict[str, Any],
        retrieve_config:Dict[str, Any],
        only_need_context
        
    ) -> str:
        """
        调用 Light RAG 的 /query 接口
        
        参数:
        query: 用户查询文本 (必需)
        mode: 查询模式 (默认 "mix")
        lightrag_config : 配置参数
        only_need_context： If True, only returns the retrieved context without generating a response.
        
        
        返回:
        str: 成功时返回响应字符串，失败时抛出异常
        
        异常:
        ValueError: 参数验证错误 (422 响应)
        RuntimeError: 其他 API 错误
        """

        base_url=lightrag_config.get("base_url")
        url = f"{base_url}/query"
        api_key :str =lightrag_config.get("api_key")

        timeout: int = lightrag_config.get("timeout",120) 
        if api_key:
            headers = {'Authorization': f'Bearer {api_key}'}
        else:
            headers = {'Content-Type': 'application/json'}

        #string:local ,global, hybrid, mix,naive ,bypass , bypass Default"mix" ,mix下,bypass不能召回上下文
        mode: str = retrieve_config.get("mode","mix")  
        
        #(boolean | null) If True, only returns the generated prompt without producing a response.
        only_need_prompt= retrieve_config.get("only_need_prompt",False) 
        # (string | null) 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.
        response_type= retrieve_config.get("response_type","Multiple Paragraphs") 
        #(integer | null) Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.
        top_k= retrieve_config.get("top_k",10) 
        # Number of text chunks to retrieve initially from vector search and keep after reranking.
        chunk_top_k= retrieve_config.get("chunk_top_k",5) 
        # (integer | null) Maximum number of tokens allocated for entity context in unified token control system.
        max_entity_tokens = retrieve_config.get("max_entity_tokens",None) 
        # (integer | null) Maximum number of tokens allocated for relationship context in unified token control system.
        max_relation_tokens = retrieve_config.get("max_relation_tokens",None) 
        # (integer | null) Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).
        max_total_tokens = retrieve_config.get("max_total_tokens",None) 
        # (array<object> | null) Stores past conversation history to maintain context. Format: [{'role': 'user/assistant', 'content': 'message'}].
        conversation_history = []
        # (integer | null) Number of complete conversation turns (user-assistant pairs) to consider in the response context.
        history_turns: int = 0
       
        # (boolean | null) Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.
        enable_rerank= retrieve_config.get("enable_rerank",True) 
        
        # 构建请求体
        payload = {"query": query, "mode": mode}
        
        # 添加可选参数
        optional_params = {
            "only_need_context": only_need_context,
            "only_need_prompt": only_need_prompt,
            "response_type": response_type,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "max_entity_tokens": max_entity_tokens,
            "max_relation_tokens": max_relation_tokens,
            "max_total_tokens": max_total_tokens,
            "conversation_history": conversation_history,
            "history_turns": history_turns,
           
            "enable_rerank": enable_rerank
        }
        
        for key, value in optional_params.items():
            if value is not None:
                payload[key] = value
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            # 处理 200 成功响应
            if response.status_code == 200:
                result=response.json()
                return result["response"]
            
            # 处理 422 验证错误
            if response.status_code == 422:
                error_details = response.json().get("detail", [])
                error_messages = []
                for error in error_details:
                    loc = ".".join(str(part) for part in error.get("loc", []))
                    msg = error.get("msg", "Unknown validation error")
                    error_messages.append(f"{loc}: {msg}")

                logg_error_message= "\n".join(error_messages)
                logger.error(f"检索 light rag error: {logg_error_message}")
                
                raise ValueError(logg_error_message)
            
            # 处理其他错误状态码
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            # 包装并抛出异常
            error_msg = f"API请求失败: {str(e)}"
            if response is not None:
                error_msg += f" | Status: {response.status_code} | Response: {response.text[:200]}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def retrieve_context(
        self,
        query: str,
        lightrag_config:Dict[str, Any],
        retrieve_config:Dict[str, Any],
    ) -> list[str]:
        reponse =self.retrieve(query,lightrag_config,retrieve_config,only_need_context=True)
       
        pattern = r'Document Chunks\(DC\)[^\n]*\n\n```json\n(.*?)\n```\n\n'
        match = re.search(pattern, reponse, re.DOTALL)
        if match:
            dc_content = match.group(1).strip()
            # 解析 JSON 数据
            try:
                data = json.loads(dc_content)
                # 提取 content 字段并形成列表
                content_list = [item['content'] for item in data]
                return content_list
            except json.JSONDecodeError as e:
                logger.error(f"retrieve_context JSON 解析错误: {e}")
        else:
            logger.warning("retrieve_context 未找到 Document Chunks(DC) 部分")
            return []

    def run_evaluation(self):
        """
        评估Dify知识库召回内容和rag工作流的回复
        """
        logger.info("开始评估lightrag 知识图谱检索和生成的质量...")

        lightrags=settings["lightrags"]

        for rag_config in lightrags:
            # 准备问答数据
            qa_excel_file=rag_config.get("qa_excel_file")
            qa_data = self.prepare_evaluation_data(qa_excel_file)
            logger.info(f"加载了 {len(qa_data)} 条测试数据")
        
            result_file=rag_config.get("result_excel_file")
            evaluation_results = []

            for retrieve_config in rag_config["retrieve_configs"]:
                retrieve_name=retrieve_config.get("name")
                # 创建RAGAS数据集
                logger.info(f"开始创建RAGAS数据集,检索策略:{retrieve_name}...")
                evaluation_dataset = self.create_evaluation_dataset(qa_data,
                                                                rag_config=rag_config,
                                                                retrieve_config=retrieve_config,
                                                                )
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
        rag_evaluator = LightRagEvaluator()
        rag_evaluator.run_evaluation()
    
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()
