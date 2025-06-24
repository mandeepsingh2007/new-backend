file_path = 'C:\python\AI-Driven-Intelligent-Trading-Assistant-for-Real-Time-Market-Analysis-and-Automated-Execution\generated_text.txt'
import pandas as pd

def analyze_text_file_sentiment(file_path):
    """
    Analyze sentiment from a small text file containing market-related information
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    results = []
    
    for line in lines:
        if len(line) < 10:
            continue
            
        result = sentiment_analyzer(line)[0]
        results.append({
            'text': line,
            'sentiment': result['label'],
            'score': result['score']
        })
    
    positive_lines = [r for r in results if r['sentiment'] == 'POSITIVE']
    positive_ratio = len(positive_lines) / len(results) if results else 0.5
    
    avg_sentiment_score = sum(r['score'] for r in results) / len(results) if results else 0.5

    if positive_ratio > 0.5:
        normalized_sentiment = (positive_ratio - 0.5) * 2
    else:
        normalized_sentiment = (positive_ratio - 0.5) * 2
    print(f"Overall Sentiment: {'POSITIVE' if positive_ratio > 0.5 else 'NEGATIVE'}")
    print(f"Positive Ratio: {positive_ratio:.2f}")
    print(f"Normalized Sentiment: {normalized_sentiment:.2f}")
    print(f"Average Sentiment Score: {avg_sentiment_score:.2f}")
    results= {
        'results': results,
        'overall_sentiment': 'POSITIVE' if positive_ratio > 0.5 else 'NEGATIVE',
        'positive_ratio': positive_ratio,
        'normalized_sentiment': normalized_sentiment,
        'avg_sentiment_score': avg_sentiment_score
    }
    return results



def generate_trading_signals(price_data, sentiment_result, risk_profile='moderate'):
    """
    Generate trading signals based on technical indicators and sentiment analysis
    """
    sentiment_score = sentiment_result['normalized_sentiment']
    
    risk_weights = {
        'conservative': {'technical': 0.8, 'sentiment': 0.2},
        'moderate': {'technical': 0.6, 'sentiment': 0.4},
        'aggressive': {'technical': 0.4, 'sentiment': 0.6}
    }
    
    weights = risk_weights.get(risk_profile, risk_weights['moderate'])
    
    signals = {}
    
    for ticker in price_data:
        ticker_data = price_data[ticker]
        
        sma_10 = ticker_data['Close'].rolling(window=10).mean().iloc[-1]
        sma_30 = ticker_data['Close'].rolling(window=30).mean().iloc[-1]
        
        technical_signal = 1 if sma_10 > sma_30 else -1
        
        combined_signal = (technical_signal * weights['technical'] + 
                           sentiment_score * weights['sentiment'])
        
        if combined_signal > 0.3:
            action = 'buy'
        elif combined_signal < -0.3:
            action = 'sell'
        else:
            action = 'hold'
        
        confidence = abs(combined_signal)
        
        signals[ticker] = {
            'action': action,
            'confidence': round(confidence, 2),
            'technical_signal': technical_signal,
            'sentiment_signal': sentiment_score,
            'combined_signal': round(combined_signal, 2)
        }
        print (f"Ticker: {ticker}, Action: {action}, Confidence: {confidence:.2f}, "
               f"Technical Signal: {technical_signal}, Sentiment Signal: {sentiment_score:.2f}, "
               f"Combined Signal: {combined_signal:.2f}")
    
    return signals

if __name__ == "__main__":
    analyze_text_file_sentiment(file_path)
    # Example usage
    file_path = 'C:\python\AI-Driven-Intelligent-Trading-Assistant-for-Real-Time-Market-Analysis-and-Automated-Execution\generated_text.txt'
    sentiment_result = analyze_text_file_sentiment(file_path)
    
    # Example price data (replace with actual data)
    price_data = {
        'AAPL': pd.DataFrame({
            'Close': [150, 152, 153, 151, 155, 157, 158]
        }),
        'GOOGL': pd.DataFrame({
            'Close': [2800, 2820, 2810, 2830, 2840, 2850, 2860]
        })
    }
    
    signals = generate_trading_signals(price_data, sentiment_result)