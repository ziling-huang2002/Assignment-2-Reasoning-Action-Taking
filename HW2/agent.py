import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from tools import search_google

load_dotenv()

# 1. 初始化 OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("找不到 API Key，請檢查 .env 檔案中是否有 OPENAI_API_KEY")

# 配置 SDK
client = OpenAI(api_key=api_key)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY 未設定，請在 .env 中加入此金鑰。")


# 2. 定義 System Prompt
SYSTEM_PROMPT = """你是一個具備邏輯推理能力的 ReAct Agent。
你的任務是透過 Thought -> Action -> Observation 的循環來解決問題。

格式規範（嚴格遵守）：
1. Thought: 
   - 簡短紀錄：簡述你在 Observation 中觀察到了什麼關鍵數據。
   - 下步規劃：思考是否需要繼續搜尋或可以開始計算。如果數據已齊全，請在此寫下純文字計算過程。
   - 計算過程：數據齊全時，用純文字橫式計算（如：2397 / 12320 = 0.194）。
2. Action: Search[精確的搜尋關鍵字] (每次只能執行一個 Action)
3. Observation: 這部分由系統提供。
4. Final Answer: 直接輸出最終答案，內容需簡短整齊。

規則：
- 務必將複雜問題拆解（Planning），例如先搜尋 A，再搜尋 B。
- 如果搜尋不到結果，請在下一個 Thought 進行反思（Reflection）並更換關鍵字。
- 如果 Observation 內容太多，請只提取對回答問題有幫助的「數字」或「事實」。
- 所有輸出請使用「繁體中文」。
- 嚴禁使用 LaTeX 或 \[ \] 符號。

---
範例 (One-shot Example):
User: 現在台北的天氣適合穿短袖嗎？
Thought: 我需要先查詢台北目前的氣溫。
Action: Search[台北 目前氣溫]
Observation: 台北現在氣溫為 28 度，天氣晴朗。
Thought: 我觀察到氣溫為 28 度，這屬於較為炎熱的天氣，非常適合穿短袖。
Final Answer: 台北目前氣溫為 28 度，氣候炎熱，非常適合穿短袖出門。
---
"""

def execute(query, max_turns=5):
    # 建立對話紀錄 (OpenAI 使用 messages 格式)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    for turn in range(max_turns):
        # ---讓 OpenAI 生成 Thought 與 Action ---
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            stop=["Observation:", "Observation"] 
        )
        
        agent_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_text})
        print(f"\n--- 第 {turn+1} 輪迭代 ---")
        print(agent_text)

        if "Final Answer:" in agent_text:
            # 只提取 Final Answer: 之後的內容，並去除前後空白
            final_result = agent_text.split("Final Answer:")[-1].strip()
            return final_result

        # --- 解析 Action ---
        action_match = re.search(r"Action: Search\[(.*?)\]", agent_text)
        
        if action_match:
            search_query = action_match.group(1)
            # 執行搜尋 (Tavily)
            search_results = search_google(search_query)
            
            # 將結果餵回給歷史紀錄
            observation_str = f"Observation: {search_results}"
            messages.append({"role": "user", "content": observation_str})
            print(f"📡 取得 Observation (長度: {len(str(search_results))})")
        
        else:
            # 如果模型沒給 Action 也沒給 Final Answer，強制要求它修正格式
            messages.append({"role": "user", "content": "Observation: 格式錯誤，請提供 Action: Search[...] 或 Final Answer。"})

    # 若超過 max_turns 仍無結果的回傳
    return "抱歉，我無法在預設步數內得出結論，請嘗試換個問法。"