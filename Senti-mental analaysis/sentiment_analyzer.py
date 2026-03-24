"""

Modern Production-Ready Sentiment Analysis Engine

Upgraded from basic VADER to state-of-the-art transformer-based models

Supports real-world text with noise, emojis, slang, etc.

"""

import os
import re
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class SentimentAnalyzer:
    """

    Production-ready sentiment analyzer using DistilBERT.

    Why DistilBERT over VADER?
    - VADER: Rule-based, limited to lexicons, struggles with context
    - DistilBERT: Neural model, understands context, trained on millions of texts
    - 40% faster than BERT, high accuracy, lightweight (268MB)
    - Handles sarcasm, negation, complex language

    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """

        Initialize sentiment analyzer with state-of-the-art model.

        Args:
            model_name: HuggingFace model ID (default: DistilBERT fine-tuned on SST-2)

        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {'GPU (CUDA)' if self.device == 0 else 'CPU'}")

        try:
            # Load transformer-based pipeline
            self.pipeline = pipeline(  # type: ignore
                "sentiment-analysis",
                model=model_name,
                device=self.device
            )
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """

        Clean and normalize text for better analysis.
        Handles emojis, URLs, mentions, extra whitespace, etc.

        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Convert to lowercase for consistency
        text = text.lower()

        # Remove URLs (http://, https://, www.)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions (@user)
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags but keep the word (e.g., #amazing -> amazing)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Normalize common emojis to text
        emoji_map = {
            '😀': ' good ', '😁': ' good ', '😂': ' funny ',
            '😍': ' love ', '❤️': ' love ', '😢': ' sad ',
            '😭': ' sad ', '😡': ' angry ', '😠': ' angry ',
            '😤': ' frustrated ', '😱': ' shock ', '🔥': ' hot ',
            '💯': ' perfect ', '👍': ' good ', '👎': ' bad ',
            '🎉': ' party ', '✨': ' amazing ',
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)

        # Remove other emojis
        text = re.sub(r'[^\w\s\.\!\?\,-]', '', text)

        # Normalize multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)

        # Normalize repeated characters
        text = re.sub(r'(\w)\1{2,}', r'\1', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _truncate_text(self, text: str, max_length: int = 512) -> str:
        """

        Truncate text to model's max token limit while preserving meaning.

        """
        # Rough estimate: 1 token ≈ 4 characters
        char_limit = max_length * 4
        if len(text) > char_limit:
            text = text[:char_limit] + "..."
        return text

    def analyze_single(self, text: str) -> Dict:
        """

        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment label, confidence, and detailed scores

        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid input: empty or non-string text")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'error': 'Invalid input'
            }

        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            if not cleaned_text:
                return {
                    'sentiment': 'NEUTRAL',
                    'confidence': 0.0,
                    'positive_score': 0.0,
                    'negative_score': 0.0,
                    'error': 'Text empty after preprocessing'
                }

            # Truncate if needed
            cleaned_text = self._truncate_text(cleaned_text)

            # Run inference with torch.no_grad() for efficiency
            with torch.no_grad():
                predictions = self.pipeline(cleaned_text)

            # Handle predictions
            if predictions and isinstance(predictions, list) and len(predictions) > 0:
                pred = predictions[0]
                if isinstance(pred, dict) and 'label' in pred and 'score' in pred:
                    label = pred['label'].upper()
                    score = pred['score']
                    if label == 'POSITIVE':
                        positive_score = score
                        negative_score = 1.0 - score
                    elif label == 'NEGATIVE':
                        negative_score = score
                        positive_score = 1.0 - score
                    else:
                        positive_score = 0.5
                        negative_score = 0.5
                else:
                    positive_score = 0.5
                    negative_score = 0.5
            else:
                positive_score = 0.5
                negative_score = 0.5

            # Determine sentiment
            if positive_score > negative_score:
                sentiment = 'POSITIVE'
                confidence = positive_score
            elif negative_score > positive_score:
                sentiment = 'NEGATIVE'
                confidence = negative_score
            else:
                sentiment = 'NEUTRAL'
                confidence = max(positive_score, negative_score)

            return {
                'text': text[:100] + "..." if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'positive_score': round(positive_score, 4),
                'negative_score': round(negative_score, 4),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'error': str(e)
            }

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """

        Analyze sentiment for multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            batch_size: Process texts in batches for speed

        Returns:
            List of sentiment analysis results

        """
        logger.info(f"Analyzing {len(texts)} texts in batches of {batch_size}")
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze_single(text) for text in batch]
            results.extend(batch_results)
            logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return results

    def get_statistics(self, results: List[Dict]) -> Dict:
        """

        Generate statistics from analysis results.

        """
        if not results:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0.0,
                'negative_pct': 0.0,
                'neutral_pct': 0.0,
                'avg_confidence': 0.0
            }

        total = len(results)
        sentiments = [r.get('sentiment', 'NEUTRAL') for r in results]
        confidences = [r.get('confidence', 0.0) for r in results]

        positive_count = sentiments.count('POSITIVE')
        negative_count = sentiments.count('NEGATIVE')
        neutral_count = sentiments.count('NEUTRAL')

        return {
            'total': total,
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'positive_pct': round(100 * positive_count / total, 2) if total > 0 else 0,
            'negative_pct': round(100 * negative_count / total, 2) if total > 0 else 0,
            'neutral_pct': round(100 * neutral_count / total, 2) if total > 0 else 0,
            'avg_confidence': round(np.mean(confidences), 4) if confidences else 0.0
        }

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()

    test_texts = [
        "The product exceeded my expectations. It's excellent!",
        "I had a terrible experience with this company.",
        "The quality is average. It meets my basic requirements.",
        "I absolutely love this product!",
        "worst purchase ever... total waste of money",
    ]

    logger.info("Starting sentiment analysis...")
    results = analyzer.analyze_batch(test_texts)

    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ANALYSIS RESULTS")
    logger.info("="*80)

    for i, result in enumerate(results, 1):
        logger.info(f"\n[{i}] {result.get('text', '')}")
        logger.info(f"    Sentiment: {result.get('sentiment')} | Confidence: {result.get('confidence')}")

    stats = analyzer.get_statistics(results)
    logger.info("\n" + "="*80)
    logger.info("STATISTICS")
    logger.info("="*80)
    logger.info(f"Total: {stats['total']}")
    logger.info(f"Positive: {stats['positive']} ({stats['positive_pct']}%)")
    logger.info(f"Negative: {stats['negative']} ({stats['negative_pct']}%)")
    logger.info(f"Neutral: {stats['neutral']} ({stats['neutral_pct']}%)")
