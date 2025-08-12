from dify_rag_evaluator import DifyRagEvaluator
from logger_config import logger
def main():
    try:
        # 执行评估
        dify_rag_evaluator = DifyRagEvaluator()
        dify_rag_evaluator.evaluate_dify_rag()
       
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")

if __name__ == "__main__":
    main()