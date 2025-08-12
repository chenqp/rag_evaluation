import os
from loguru import logger
import sys


def setup_logger():
    """
    统一配置Loguru日志系统
    
    日志配置遵循以下规范：
    1. 日志级别可通过环境变量LOG_LEVEL配置，默认为INFO
    2. 日志文件路径可通过环境变量LOG_PATH配置，默认为logs目录
    3. 日志文件大小限制可通过环境变量LOG_ROTATION配置，默认为50 MB
    4. 日志格式统一包含时间戳、日志级别和日志信息
    5. 同时支持输出到控制台和文件
    """
    # 移除默认的日志配置
    logger.remove()
    
    # 从环境变量获取配置参数，支持通过环境变量覆盖默认值
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_path = os.getenv('LOG_PATH', 'logs')
    log_rotation = os.getenv('LOG_ROTATION', '50 MB')
    
    # 确保日志目录存在
    os.makedirs(log_path, exist_ok=True)
    
    # 配置日志格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件日志处理器（带时间戳）
    logger.add(
        f"{log_path}/ragtest_{{time}}.log",
        rotation=log_rotation,
        encoding="utf-8",
        level=log_level,
        format=log_format
    )
    
    # 添加文件日志处理器（当前日志）
    logger.add(
        f"{log_path}/ragtest.log",
        rotation=log_rotation,
        encoding="utf-8",
        level=log_level,
        format=log_format
    )
    
    # 添加控制台日志处理器
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format
    )
    
    return logger


# 初始化全局logger实例
logger = setup_logger()