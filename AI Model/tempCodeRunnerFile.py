file_path = 'C:\python\AI-Driven-Intelligent-Trading-Assistant-for-Real-Time-Market-Analysis-and-Automated-Execution\generated_text.txt'
def analyze_text_file_sentiment(file_path):
    """
    Analyze sentiment from a small text file containing market-related information
    """
    # Load the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Clean the lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Use a pre-trained sentiment analysis model
    from transformers import pipeline
    
    # Load sentiment analyzer - using distilbert for speed
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Process the text in batches (if lines are short, combine some)
    results = []
    
    # Option 1: Process each line separately
    for line in lines:
        # Skip very short lines or headers
        if len(line) < 10:
            continue
            
        result = sentiment_analyzer(line)[0]
        results.append({
            'text': line,
            'sentiment': result['label'],
            'score': result['score']
        })
    
    # Calculate overall sentiment
    positive_lines = [r for r in results if r['sentiment'] == 'POSITIVE']
    positive_ratio = len(positive_lines) / len(results) if results else 0.5
    
    avg_sentiment_score = sum(r['score'] for r in results) / len(results) if results else 0.5
    
    # Normalize to [-1, 1] scale where 1 is fully positive
    if positive_ratio > 0.5:
        # More positive than negative, so scale from [0.5, 1]
        normalized_sentiment = (positive_ratio - 0.5) * 2
    else:
        # More negative than positive, so scale from [-1, 0]
        normalized_sentiment = (positive_ratio - 0.5) * 2
    
    return {
        'results': results,
        'overall_sentiment': 'POSITIVE' if positive_ratio > 0.5 else 'NEGATIVE',
        'positive_ratio': positive_ratio,
        'normalized_sentiment': normalized_sentiment,
        'avg_sentiment_score': avg_sentiment_score
    }

analyze_text_file_sentiment(file_path)

