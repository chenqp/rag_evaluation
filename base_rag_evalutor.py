from abc import ABC, abstractmethod

import asyncio
import pandas as pd

from logger_config import logger
from typing import List, Dict, Any
import re
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper


class BaseRagEvaluator(ABC):
    def __init__(self):
        asyncio.run(self.initialize_evaluator())
    
    async def initialize_evaluator(self):
        """初始化评估器，包括设置语言适配"""
        
        # 允许子类自定义指标
        self.metrics, evaluator_llm= self.get_metrics()
        logger.info("正在修改评估prompt适配中文...")
        # 适配所有指标的中文prompt
        for metric in self.metrics:
            await metric.adapt_prompts(language='chinese', llm=evaluator_llm)
        
        logger.info("评估prompt语言已适配为中文")
    @abstractmethod
    def get_metrics(self) -> tuple[ list,LangchainLLMWrapper]:
        """子类必须实现的抽象方法，返回评估指标列表"""
        pass

    def prepare_evaluation_data(self, excel_file: str) -> List[Dict[str, Any]]:
        """从Excel准备评估数据（通用实现）"""
        df = pd.read_excel(excel_file)
        return [{
            "question": row.get("问题", ""),
            "ground_truth": row.get("答案", "")
        } for _, row in df.iterrows()]

    @staticmethod
    def clean_rag_answer(answer: str) -> str:
        """清理RAG答案中的特殊标签（通用实现）"""
        if not answer:
            return ""
        
        # 移除各种工具调用标签
        patterns = [
            r'<tool_call>[\s\S]*?</tool_call>',
            r'<think>[\s\S]*?</think>',
            r'think:\s*',
            r'<.*?>'
        ]
        
        for pattern in patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        return answer.strip()
    
    @abstractmethod
    def create_evaluation_dataset(self, qa_data: List[Dict[str, Any]], 
                                 rag_config: Dict[str, Any],**kwargs) -> EvaluationDataset:
        """创建评估数据集（子类实现）"""
        pass

    @abstractmethod
    def run_evaluation(self):
        """执行评估流程（子类实现）"""
        pass

    def save_results(self, evaluation_results, result_file: str):
        """保存评估结果到Excel（通用实现）"""
        if isinstance(evaluation_results, list):
            with pd.ExcelWriter(result_file,engine='xlsxwriter') as writer:
                for result in evaluation_results:
                    result_df = result['result_df']
                    retrieve_name = result['retrieve_name']
                    result_df.to_excel(writer, index=False,sheet_name=retrieve_name)
        else:
            evaluation_results.to_pandas().to_excel(result_file, index=False)
        
        logger.info(f"评估结果已保存到 {result_file}")