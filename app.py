import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath("D:/Projects/Trade-AI/TradeScribe-AI/ai_agents")))

# Import agents
from ai_agents.yfinance_agent import fetch_stock_data
from ai_agents.gemini_agent import get_news_summary
from ai_agents.historical_analysis_agent import historical_stock_analysis

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/api/stocks", methods=["POST"])
def handle_stock():
    print("📥 Incoming request received...")

    try:
        data = request.get_json()
        print("✅ Request JSON parsed:", data)

        stock = data.get("stock")
        symbol = data.get("symbol")
        agent = data.get("agent", "all")
        risk_level = data.get("riskLevel")
        timeline = data.get("timeline")

        if not stock or not symbol:
            print("❌ Missing stock or symbol")
            return jsonify({"error": "Missing 'stock' or 'symbol'"}), 400

        response = {}

        # Agent: yfinance
        if agent in ["all", "yfinance"]:
            try:
                print("🔍 Fetching data from yfinance agent...")
                df = fetch_stock_data(stock)

                if df is not None:
                    df = df.tail(5).reset_index()
                    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # Format date
                    response["stock_data"] = df.to_dict(orient="records")
                    print("✅ yfinance data processed with date")
                else:
                    response["stock_data"] = []
                    print("⚠️ yfinance returned empty dataframe")
            except Exception as e:
                print("❌ Error in yfinance_agent:", e)
                response["stock_data_error"] = str(e)

        # Agent: gemini
        if agent in ["all", "gemini"]:
            try:
                print("🔍 Fetching news summary from gemini agent...")
                news = get_news_summary(stock, symbol)
                print("✅ Gemini news summary received")
                response["news_summary"] = news
            except Exception as e:
                print("❌ Error in gemini_agent:", e)
                response["news_summary_error"] = str(e)

        # Agent: historical analysis
        if agent in ["all", "historical"]:
            try:
                print("🔍 Running historical analysis agent...")
                analysis = historical_stock_analysis(symbol)

                # If it failed or lacks recommendations, fallback
                if analysis.get("error") or not analysis.get("top_recommendations"):
                    print("⚠️ Historical agent failed or no data, injecting fallback")
                    analysis = {
                        "top_recommendations": [
                            {
                                "symbol": symbol,
                                "company": stock,
                                "recommendation": "Hold",
                                "confidence": 70,
                                "price_target": 1500.00,
                                "trend": "neutral"
                            }
                        ],
                        "error": "Insufficient data, showing fallback recommendation"
                    }

                response["historical_analysis"] = analysis
                print("✅ Historical analysis added to response")

            except Exception as e:
                print("❌ Error in historical_analysis_agent:", e)
                response["historical_analysis_error"] = str(e)

        print("✅ Final API Response:", response)
        return jsonify(response), 200

    except Exception as e:
        print("❌ General error in /api/stocks:", e)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 Flask app is starting...")
    app.run(debug=True)
