from openai import OpenAI
import json
from datetime import datetime

class TimeParser:
    def __init__(self, api_key: str):
        """
        初始化时间解析器，对接 DeepSeek API。
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

    def extract_time_constraints(self, query: str):
        """
        利用 DeepSeek 的语义能力，将模糊的时间词（如“去年夏天”）转为具体的日期范围。
        """
        # 1. 快速过滤：如果连一个跟时间相关的字都没有，直接返回空，省点 API 钱
        time_keywords = ["年", "月", "天", "去", "前", "后", "最近", "夏", "冬", "春", "秋", "时候", "刚"]
        if not any(word in query for word in time_keywords):
            return {"start_date": None, "end_date": None}

        # 2. 获取当前日期作为解析基准（非常重要，否则 AI 不知道“去年”是哪年）
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = (
            "你是一个精准的时间解析专家。用户会提供一段查询描述，你需要提取其中的时间范围约束。"
            f"已知今天的日期是 {current_date}。请基于此日期解析相对时间。"
            
            "季节定义：春(3-5月), 夏(6-8月), 秋(9-11月), 冬(12-2月)。"
            "必须返回标准的 JSON 格式，包含 start_date 和 end_date 两个字段（格式为 YYYY-MM-DD）。"
            "如果没有提到明确的时间范围，请将这两个字段设为 null。"
            "注意：如果用户说'XX年'（如22年），通常指'20XX年'。如果是'22年以前'，请将 start_date 设为 null，将 end_date 设为 2021-12-31。"
        )

        user_prompt = f"查询内容：'{query}'"

        try:
            # 3. 调用 DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # 强制要求返回 JSON
                temperature=0 # 设为 0 保证结果最稳定
            )
            
            # 4. 解析结果
            result = json.loads(response.choices[0].message.content)
            return {
                "start_date": result.get("start_date"),
                "end_date": result.get("end_date")
            }
        except Exception as e:
            # 万一 API 挂了或者余额不足，返回空，不影响主程序运行
            print(f"⚠️ TimeParser 解析失败: {e}")
            return {"start_date": None, "end_date": None}