---
title: "LangChain 1.0 React Agent Demo：构建智能推理代理"
date: 2025-12-30T15:00:06+08:00
draft: false

keywords: ["LangChain", "React Agent", "Python", "AI Agent", "LLM"]
summary: "基于 LangChain 1.0 版本，使用 Python 开发一个 React（Reasoning and Acting）模式的智能代理 demo，展示如何构建能够推理和行动的 AI Agent"
---

# LangChain 1.0 React Agent Demo：构建智能推理代理

LangChain 1.0 版本带来了许多重要的改进和新特性，其中 React（Reasoning and Acting）模式的 Agent 是一个强大的功能。本文将带你从零开始，使用 Python 构建一个完整的 React Agent Demo。

## 什么是 React Agent？

React（Reasoning and Acting）是一种结合了推理（Reasoning）和行动（Acting）的 Agent 模式。Agent 会：

1. **推理（Reasoning）**：分析问题，思考需要采取什么行动
2. **行动（Acting）**：调用工具执行具体操作
3. **观察（Observing）**：观察工具执行的结果
4. **循环**：基于观察结果继续推理，直到完成任务

这种模式让 Agent 能够处理复杂的多步骤任务，通过工具调用来获取信息、执行操作，最终给出答案。

## 环境准备

### 1. Python 版本要求

**重要**：LangChain 1.0 要求 Python 版本 **≥ 3.10**。Python 3.9 及以下版本不再支持。

检查你的 Python 版本：

```bash
python --version
# 或
python3 --version
```

如果版本低于 3.10，请先升级 Python 版本。

### 2. 安装依赖

安装必要的 Python 包：

```bash
pip install langchain>=1.0.0
pip install langchain-openai
pip install langchain-community
pip install langchain-experimental
pip install langchainhub
```

### 3. 配置环境变量

设置你的 DeepSeek API 密钥：

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

如果需要使用搜索工具，还需要设置 Tavily API Key：

```bash
export TAVILY_API_KEY="your-tavily-api-key-here"
```

## 完整代码实现

### 基础版本：简单的 React Agent

让我们先创建一个基础的 React Agent：

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# 初始化语言模型（使用 DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 定义工具
tools = [
    DuckDuckGoSearchRun()
]

# 从 LangChain Hub 加载 React prompt
prompt = hub.pull("hwchase17/react")

# 创建 React Agent
agent = create_react_agent(llm, tools, prompt)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 运行 Agent
if __name__ == "__main__":
    response = agent_executor.invoke({
        "input": "什么是 LangChain？它有哪些主要特性？"
    })
    print("\n最终答案：")
    print(response["output"])
```

### 进阶版本：多工具 React Agent

让我们创建一个更强大的版本，包含多个工具：

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from datetime import datetime

# 初始化语言模型（使用 DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 定义自定义工具：获取当前时间
def get_current_time(query: str) -> str:
    """获取当前日期和时间"""
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# 定义自定义工具：计算器
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# 创建工具列表
tools = [
    DuckDuckGoSearchRun(name="search", description="搜索互联网获取最新信息"),
    PythonREPLTool(name="python_repl", description="执行 Python 代码"),
    Tool(
        name="get_time",
        func=get_current_time,
        description="获取当前日期和时间"
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="执行数学计算，输入数学表达式"
    ),
]

# 从 LangChain Hub 加载 React prompt
prompt = hub.pull("hwchase17/react")

# 创建 React Agent
agent = create_react_agent(llm, tools, prompt)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=60
)

def run_agent(query: str):
    """运行 Agent 并返回结果"""
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"执行错误：{str(e)}"

if __name__ == "__main__":
    # 示例 1：搜索信息
    print("=" * 50)
    print("示例 1：搜索信息")
    print("=" * 50)
    result1 = run_agent("LangChain 1.0 版本有哪些新特性？")
    print(f"\n结果：{result1}\n")
    
    # 示例 2：数学计算
    print("=" * 50)
    print("示例 2：数学计算")
    print("=" * 50)
    result2 = run_agent("计算 (1234 * 5678) / 100 的结果")
    print(f"\n结果：{result2}\n")
    
    # 示例 3：Python 代码执行
    print("=" * 50)
    print("示例 3：Python 代码执行")
    print("=" * 50)
    result3 = run_agent("使用 Python 生成一个包含 10 个随机数的列表，并计算它们的平均值")
    print(f"\n结果：{result3}\n")
    
    # 示例 4：综合任务
    print("=" * 50)
    print("示例 4：综合任务")
    print("=" * 50)
    result4 = run_agent("今天是几号？然后搜索一下今天有什么重要的科技新闻")
    print(f"\n结果：{result4}\n")
```

### 高级版本：带记忆的 React Agent

为了让 Agent 能够记住对话历史，我们可以添加记忆功能：

```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 初始化语言模型（使用 DeepSeek）
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 定义工具
tools = [DuckDuckGoSearchRun()]

# 加载 React prompt
prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# 存储对话历史的字典
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 创建带历史的 Agent
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    session_id = "demo_session"
    
    # 第一轮对话
    print("=" * 50)
    print("第一轮对话")
    print("=" * 50)
    response1 = agent_with_history.invoke(
        {"input": "我的名字是张三"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Agent: {response1['output']}\n")
    
    # 第二轮对话（Agent 应该记得名字）
    print("=" * 50)
    print("第二轮对话")
    print("=" * 50)
    response2 = agent_with_history.invoke(
        {"input": "我的名字是什么？"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Agent: {response2['output']}\n")
```

## React Agent 的工作原理

### 1. Prompt 结构

React Agent 使用的 prompt 模板包含以下部分：

- **系统指令**：告诉 Agent 如何思考和行动
- **工具描述**：列出可用的工具及其使用方法
- **示例**：展示如何格式化和使用工具
- **思考-行动-观察循环**：指导 Agent 的执行流程

### 2. 执行流程

```
用户输入
    ↓
Agent 推理（思考需要做什么）
    ↓
选择工具并调用
    ↓
观察工具执行结果
    ↓
基于结果继续推理
    ↓
（如果需要）再次调用工具
    ↓
给出最终答案
```

### 3. 工具调用格式

Agent 使用特定的格式来调用工具：

```
Thought: 我需要搜索信息
Action: search
Action Input: LangChain 1.0 features
Observation: [工具返回的结果]
Thought: 基于搜索结果，我可以给出答案
Final Answer: LangChain 1.0 的主要特性包括...
```

## 实际应用场景

### 1. 信息检索和问答

Agent 可以搜索最新信息并回答用户问题：

```python
query = "2024 年 AI 领域有哪些重要突破？"
response = agent_executor.invoke({"input": query})
```

### 2. 数据分析

结合 Python REPL 工具，Agent 可以执行数据分析任务：

```python
query = "分析这个 CSV 文件，找出销售额最高的产品类别"
response = agent_executor.invoke({"input": query})
```

### 3. 代码生成和执行

Agent 可以生成代码并执行：

```python
query = "写一个 Python 函数来计算斐波那契数列的第 n 项，并测试 n=10 的情况"
response = agent_executor.invoke({"input": query})
```

### 4. 多步骤任务规划

Agent 可以规划并执行复杂的多步骤任务：

```python
query = "先搜索今天的天气，然后根据天气情况推荐合适的活动"
response = agent_executor.invoke({"input": query})
```

## 最佳实践

### 1. 工具设计

- **清晰的描述**：为每个工具提供清晰、详细的描述
- **合适的粒度**：工具应该专注于单一功能
- **错误处理**：确保工具能够优雅地处理错误

### 2. Prompt 优化

- **明确的指令**：告诉 Agent 何时使用工具，何时给出最终答案
- **示例引导**：提供好的示例来引导 Agent 的行为
- **约束设置**：设置合理的迭代次数和执行时间限制

### 3. 错误处理

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=10,            # 限制最大迭代次数
    max_execution_time=60,        # 限制最大执行时间
    return_intermediate_steps=True  # 返回中间步骤
)
```

### 4. 调试技巧

- 启用 `verbose=True` 查看详细的执行过程
- 使用 `return_intermediate_steps=True` 获取中间步骤
- 检查工具的输出是否符合预期

## 常见问题

### 1. Agent 陷入循环

**问题**：Agent 反复调用同一个工具，无法完成任务

**解决方案**：
- 设置 `max_iterations` 限制迭代次数
- 改进工具描述，让 Agent 更清楚何时停止
- 优化 prompt，明确告诉 Agent 何时给出最终答案

### 2. 工具调用失败

**问题**：Agent 调用工具时出现错误

**解决方案**：
- 检查工具的参数格式是否正确
- 添加 `handle_parsing_errors=True` 处理解析错误
- 在工具函数中添加错误处理逻辑

### 3. 成本控制

**问题**：Agent 调用 LLM 和工具产生较高成本

**解决方案**：
- 设置 `max_iterations` 限制
- 使用更便宜的模型（如 deepseek-chat）
- 缓存工具调用结果
- 优化 prompt 减少不必要的推理步骤

## 总结

LangChain 1.0 的 React Agent 提供了一个强大而灵活的框架来构建智能代理。通过结合推理和行动，Agent 能够：

- 理解复杂的用户需求
- 规划多步骤任务
- 调用工具获取信息或执行操作
- 基于结果进行推理并给出答案

本文提供了从基础到高级的完整示例，你可以根据自己的需求进行扩展和定制。React Agent 的强大之处在于它的灵活性——通过添加不同的工具，你可以构建出各种不同用途的智能代理。

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [React Agent 论文](https://arxiv.org/abs/2210.03629)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

---

希望这个 Demo 能帮助你理解和使用 LangChain 1.0 的 React Agent。如果你有任何问题或建议，欢迎交流讨论！
