# ai_agents/gemini_agent.py

import google.generativeai as genai

def get_news_summary(stock_name: str, ticker_symbol: str) -> str:
    API_KEY = "AIzaSyCw7R-FDCBts6kZF8F0rF5UVmgQljJ-dkM"
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = (
        f"Find all financial and stock-related news about {stock_name} ({ticker_symbol}) from the past 30 days. "
        "Summarize major developments, financial reports, partnerships, management changes, regulations, or trends."
    )

    response = model.generate_content(prompt)
    with open("generated_text.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

    return response.text
