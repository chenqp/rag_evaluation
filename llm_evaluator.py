from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from config import settings
from logger_config import logger
def get_evaluator_llm() -> LangchainLLMWrapper:
    llm_config=settings.get("evaluator_llm")
    openai_api_key = llm_config.get("api_key")
    openai_base_url = llm_config.get("base_url").rstrip('/')
    openai_model = llm_config.get("model")
    temperature = llm_config.get("temperature",0.6)

    # 创建自定义LLM，支持DeepSeek等OpenAI兼容的API
    logger.info(f"使用LLM进行评估: {openai_model}")
    
    custom_llm = ChatOpenAI(
        model=openai_model,
        openai_api_key=openai_api_key,
        openai_api_base=openai_base_url,  # 支持DeepSeek等OpenAI兼容API
        temperature=temperature
    )
    # 包装为评估大模型
    evaluator_llm = LangchainLLMWrapper(custom_llm)
    return evaluator_llm