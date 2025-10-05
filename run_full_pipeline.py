# run_pipeline.py
# Main pipeline script to run the entire workflow

import os
import sys

def run_step(step_name, script_name):
    """Run a pipeline step and check for errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}\n")

    result = os.system(f"python {script_name}")

    if result != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {result}")
        return False
    else:
        print(f"\n[SUCCESS] Step '{step_name}' completed successfully")
        return True

def main():
    print("[INFO] Starting FinBERT-LSTM Stock Prediction Pipeline")
    print("[INFO] This will run all steps from data collection to model training")

    steps = [
        ("Stock Data Collection", "collect_stock_data.py"),
        ("News Data Collection", "collect_news_rss.py"),
        ("Data Cleaning", "clean_news_data.py"),
        ("Sentiment Analysis", "analyze_sentiment.py"),
        ("MLP Model Training", "train_model_mlp.py"),
        ("LSTM Model Training", "train_model_lstm.py"),
        ("FinBERT-LSTM Model Training", "train_model_finbert_lstm.py"),
    ]

    for step_name, script_name in steps:
        if not os.path.exists(script_name):
            print(f"[WARNING] Script {script_name} not found, skipping...")
            continue

        success = run_step(step_name, script_name)

        if not success:
            print(f"\n[ERROR] Pipeline stopped at step: {step_name}")
            print(f"[INFO] Please fix the error and run again")
            sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n[INFO] Check the output files:")
    print("  - stock_price.csv: Stock price data")
    print("  - news.csv: Raw news data")
    print("  - news_data.csv: Cleaned news data")
    print("  - sentiment.csv: Sentiment scores")
    print("\n[INFO] Model performance metrics have been printed above")

if __name__ == "__main__":
    main()
