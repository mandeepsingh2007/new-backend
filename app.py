import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow CORS (important for frontend integration)
app = Flask(__name__)
CORS(app)

# Add local directories to sys.path for import resolution
sys.path.append(os.path.join(os.path.dirname(__file__), "ai_agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "AI Model"))

# Import custom AI agents and prediction pipeline
from ai_agents.yfinance_agent import fetch_stock_data
from ai_agents.gemini_agent import get_news_summary
from ai_agents.historical_analysis_agent import historical_stock_analysis
from pipeline import run_stock_prediction

@app.route("/api/stocks", methods=["POST"])
def handle_stock():
    print("📥 Incoming request received...")

    try:
        data = request.get_json()
        print("✅ Parsed request:", data)

        stock = data.get("stock")
        symbol = data.get("symbol")
        agent = data.get("agent", "all")
        risk_level = data.get("riskLevel")
        timeline = data.get("timeline")

        if not stock or not symbol:
            return jsonify({"error": "Missing 'stock' or 'symbol'"}), 400

        response = {}

        # --- YFinance Agent ---
        if agent in ["all", "yfinance"]:
            try:
                print("📊 Fetching stock data from yfinance...")
                df = fetch_stock_data(stock)
                response["stock_data"] = df.tail(5).to_dict(orient="records") if df is not None else []
            except Exception as e:
                print("❌ yfinance error:", e)
                response["stock_data_error"] = str(e)

        # --- Gemini Agent ---
        if agent in ["all", "gemini"]:
            try:
                print("📰 Fetching news summary...")
                news = get_news_summary(stock, symbol)
                response["news_summary"] = news
            except Exception as e:
                print("❌ Gemini error:", e)
                response["news_summary_error"] = str(e)

        # --- Historical Analysis Agent ---
        if agent in ["all", "historical"]:
            try:
                print("📈 Running historical analysis...")
                analysis = historical_stock_analysis(symbol)
                response["historical_analysis"] = analysis
            except Exception as e:
                print("❌ Historical analysis error:", e)
                response["historical_analysis_error"] = str(e)

        # --- LSTM Prediction Pipeline ---
        if agent in ["all", "lstm"]:
            try:
                print("🧠 Running LSTM prediction pipeline...")
                result = run_stock_prediction(symbol)

                if result:
                    future_df = result.get("future_df")
                    trend = result.get("trend_analysis")

                    response["forecast_data"] = future_df.tail(5).to_dict(orient="records") if future_df is not None else []
                    response["trend"] = trend
                    print("✅ Forecast and trend added.")
                else:
                    response["forecast_error"] = "No result returned from prediction pipeline"
            except Exception as e:
                print("❌ LSTM pipeline error:", e)
                response["forecast_error"] = str(e)

        print("✅ Final response ready.")
        return jsonify(response), 200

    except Exception as e:
        print("❌ Server error:", e)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    print("🚀 Flask app starting...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
