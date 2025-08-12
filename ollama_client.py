import requests
import json
import os
from typing import Optional, Dict, List

class OllamaClient:
    """
    Ollama客户端，用于与Ollama API交互，获取单轮和多轮对话结果
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama服务的基础URL，默认从环境变量OLLAMA_BASE_URL获取，否则使用"http://localhost:11434"
            model: 使用的模型名称，默认从环境变量OLLAMA_MODEL获取，否则使用"llama3"
        """
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip('/')
        self.model = model or os.getenv("OLLAMA_MODEL")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                         temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """
        生成单轮对话响应
        
        Args:
            prompt: 用户输入的提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成token数
            
        Returns:
            模型生成的响应文本
        """
        url = f"{self.base_url}/api/generate"
        
        # 构建请求载荷
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # 如果提供了系统提示，则添加到载荷中
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # 发送POST请求到Ollama API
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求Ollama API时出错: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"解析Ollama响应时出错: {str(e)}")
    
    def chat(self, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """
        使用聊天接口进行对话
        
        Args:
            messages: 消息列表，格式为[{"role": "user/system/assistant", "content": "消息内容"}, ...]
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成token数
            
        Returns:
            模型生成的响应文本
        """
        url = f"{self.base_url}/api/chat"
        
        # 构建请求载荷
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            # 发送POST请求到Ollama API
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求Ollama聊天API时出错: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"解析Ollama聊天响应时出错: {str(e)}")

class OllamaConversation:
    """
    Ollama多轮对话管理类，用于维护对话历史并进行多轮对话
    """
    
    def __init__(self, client: OllamaClient = None, system_prompt: str = None):
        """
        初始化对话管理器
        
        Args:
            client: Ollama客户端实例，如果未提供则创建一个新的
            system_prompt: 系统提示，将在每轮对话中使用
        """
        self.client = client or OllamaClient()
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        
        # 如果提供了系统提示，添加到历史记录中
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
    
    def add_user_message(self, message: str):
        """
        添加用户消息到对话历史
        
        Args:
            message: 用户消息内容
        """
        self.history.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """
        添加助手消息到对话历史
        
        Args:
            message: 助手消息内容
        """
        self.history.append({"role": "assistant", "content": message})
    
    def get_response(self, user_message: str = None, temperature: float = 0.3, 
                     max_tokens: int = 4096) -> str:
        """
        获取对话响应
        
        Args:
            user_message: 用户消息，如果提供则添加到历史记录中
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Returns:
            助手的响应消息
        """
        # 如果提供了用户消息，则添加到历史记录中
        if user_message:
            self.add_user_message(user_message)
        
        # 获取模型响应
        response = self.client.chat(self.history, temperature=temperature, max_tokens=max_tokens)
        
        # 将响应添加到历史记录中
        self.add_assistant_message(response)
        
        return response
    
    def reset(self):
        """
        重置对话历史
        """
        self.history.clear()
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史记录
        """
        return self.history.copy()

def get_single_turn_response(question: str, model: str = None, 
                           base_url: str = None,
                           temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """
    获取单轮对话结果的便捷函数
    
    Args:
        question: 用户问题
        model: 使用的模型名称，默认从环境变量获取
        base_url: Ollama服务的基础URL，默认从环境变量获取
        
    Returns:
        模型生成的回答
    """
    client = OllamaClient(base_url=base_url, model=model,temperature=temperature,max_tokens=max_tokens)
    
    # 使用聊天接口进行单轮对话
    messages = [
        {"role": "user", "content": question}
    ]
    
    return client.chat(messages)

def get_multi_turn_response(conversation_history: List[Dict[str, str]], 
                          model: str = None, base_url: str = None,
                          temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    获取多轮对话结果的便捷函数
    
    Args:
        conversation_history: 对话历史，格式为[{"role": "user/system/assistant", "content": "消息内容"}, ...]
        model: 使用的模型名称，默认从环境变量获取
        base_url: Ollama服务的基础URL，默认从环境变量获取
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        模型生成的回答
    """
    client = OllamaClient(base_url=base_url, model=model)
    return client.chat(conversation_history, temperature=temperature, max_tokens=max_tokens)

def start_conversation(system_prompt: str = None, model: str = None, 
                      base_url: str = None) -> OllamaConversation:
    """
    启动一个多轮对话会话
    
    Args:
        system_prompt: 系统提示
        model: 使用的模型名称，默认从环境变量获取
        base_url: Ollama服务的基础URL，默认从环境变量获取
        
    Returns:
        OllamaConversation实例
    """
    client = OllamaClient(base_url=base_url, model=model)
    return OllamaConversation(client, system_prompt)

# 统一的对话接口
def chat_with_ollama(messages: List[Dict[str, str]], model: str = None, 
                    base_url: str = None, temperature: float = 0.7, 
                    max_tokens: int = 2048) -> str:
    """
    通用的Ollama对话接口，支持单轮和多轮对话
    
    Args:
        messages: 消息列表，格式为[{"role": "user/system/assistant", "content": "消息内容"}, ...]
        model: 使用的模型名称，默认从环境变量获取
        base_url: Ollama服务的基础URL，默认从环境变量获取
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        模型生成的回答
    """
    client = OllamaClient(base_url=base_url, model=model)
    return client.chat(messages, temperature=temperature, max_tokens=max_tokens)

def run_examples():
    """
    运行示例代码，演示如何使用Ollama对话功能
    """
    print("=== Ollama 对话功能示例 ===")
    
    # 单轮对话示例
    try:
        print("\n1. 单轮对话示例:")
        question = "你好，介绍一下你自己"
        response = get_single_turn_response(question)
        print(f"问题: {question}")
        print(f"回答: {response}")
        print("-" * 50)
        
        # 使用统一接口进行单轮对话
        messages = [{"role": "user", "content": "什么是人工智能？"}]
        response = chat_with_ollama(messages)
        print(f"统一接口单轮对话: {response}")
        print("-" * 50)
        
        # 多轮对话示例
        print("\n2. 多轮对话示例:")
        conversation = start_conversation("你是一个有帮助的AI助手")
        
        # 第一轮对话
        user_input = "我喜欢编程"
        response = conversation.get_response(user_input)
        print(f"用户: {user_input}")
        print(f"助手: {response}")
        
        # 第二轮对话
        user_input = "我刚才说了什么？"
        response = conversation.get_response(user_input)
        print(f"用户: {user_input}")
        print(f"助手: {response}")
        
        # 使用统一接口进行多轮对话
        history = conversation.get_history()
        follow_up = [{"role": "user", "content": "我的名字是什么？"}]
        full_history = history + follow_up
        response = chat_with_ollama(full_history)
        print(f"统一接口多轮对话: {response}")
        
        print("-" * 50)
        print("完整对话历史:")
        for msg in conversation.get_history():
            print(f"{msg['role']}: {msg['content']}")
            
    except Exception as e:
        print(f"错误: {e}")

# 示例用法
if __name__ == "__main__":
    pass