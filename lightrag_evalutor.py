import llm_evaluator
from ragas.metrics import (
    AnswerAccuracy,
)
from typing import List, Dict, Any
from config import settings
from logger_config import logger
import pandas as pd
from ragas import SingleTurnSample, EvaluationDataset
import ollama_client
import re
from ragas import evaluate
import asyncio

class LightRagEvaluator:
    def __init__(self):
       asyncio.run(self.set_adapt_prompts())
    async def set_adapt_prompts(self):
        evaluator_llm=llm_evaluator.get_evaluator_llm()
        self.llm = evaluator_llm
        answer_accuracy=AnswerAccuracy(llm=evaluator_llm)
        await answer_accuracy.adapt_prompts(language='chinese',llm=evaluator_llm)

        # 定义评估指标
        self.metrics = [
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
    
    def create_evaluation_dataset(self,qa_data: List[Dict[str, Any]], 
                                  lightrag_config:Dict[str, Any],
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
            
            # 通过RAG获取答案
            rag_response = self.get_rag_answer(query,lightrag_config)
            
            sample= SingleTurnSample(
                user_input=query,
                response=rag_response,
                reference=str(ground_truth)
            )
            samples.append(sample)
            logger.info(f"RAG系统生成的答案: {rag_response[:200]}..." if rag_response else "无答案")
        
        # 创建评估数据集
        evaluation_dataset = EvaluationDataset(samples)
        
        return evaluation_dataset

    def get_rag_answer(self,question: str,lightrag_config:Dict[str, Any],) -> str:
        """
        通过ollama兼容接口获取ligthrag的answer
        Args:
            question: 用户提问
            ollama_client: ollama兼容客户端
            
        Returns:
            lightRAG系统生成的答案
        """
        try:
            # 使用ollma客户端调用API
            base_url= lightrag_config.get("base_url")
            model = lightrag_config.get("model")
            temperature=lightrag_config.get("temperature",0.3)
            max_tokens=lightrag_config.get("max_tokens",4096)
            answer=ollama_client.get_single_turn_response(question,model=model,base_url=base_url,
                                                          temperature=temperature,max_tokens=max_tokens)
            
            # 去除answer中的多余标签
            answer = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', answer)  # Handle multiline with [\s\S]
            answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
            answer = re.sub(r'think:\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'<.*?>', '', answer).strip()
            
            return answer
        except Exception as e:
            logger.error(f"获取RAG答案失败: {e}")
            return ""
    def evaluate_lightrag(self):
        """
        评估Dify知识库召回内容和rag工作流的回复
        """
        logger.info("开始评估lightrag生成的质量...")

        lightrags=settings["lightrags"]

        for lightrag in lightrags:
            # 准备问答数据
            qa_excel_file=lightrag.get("qa_excel_file")
            qa_data = self.prepare_evaluation_data(qa_excel_file)
            logger.info(f"加载了 {len(qa_data)} 条测试数据")
        
            result_file=lightrag("result_excel_file")

            evaluation_dataset = self.create_evaluation_dataset(qa_data,
                                                                lightrag_config=lightrag)
            # 执行评估
            logger.info("开始执行RAGAS评估lightrag...")
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
            )
        
            logger.info("lightrag评估完成")
            
            # 保存结果
            result_df = result.to_pandas()
                
            with pd.ExcelWriter(result_file,engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False)
            logger.info(f"评估结果已保存到{result_file}")

def main():
    try:
        # 执行评估
        rag_evaluator = LightRagEvaluator()
        rag_evaluator.evaluate_lightrag()
       
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()
