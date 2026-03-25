from agent import execute
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("=== ReAct Agent 已啟動 (輸入'exit'結束對話) ===")
    
    while True:
        user_input = input("\nQuestion：")
        
        # 檢查是否要結束
        if user_input.lower() in ['exit']:
            print("對話結束，再見！")
            break
            
        if not user_input.strip():
            continue

        result = execute(user_input)
        
        print("\n" + "="*20 + " Final Answer " + "="*20)
        print(result)
        print("="*54)