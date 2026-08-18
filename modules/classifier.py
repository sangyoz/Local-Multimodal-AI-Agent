# modules/classifier.py
from modules.llm import chat

def classify_paper(text: str, topics: list[str]) -> str:
    prompt = f"""
请根据以下论文内容，从给定主题中选择一个最相关的主题。

候选主题：{", ".join(topics)}

论文内容（节选）：
{text[:3000]}

要求：
- 只返回主题名
- 不要解释
"""
    result = chat(prompt)
    return result.strip()
