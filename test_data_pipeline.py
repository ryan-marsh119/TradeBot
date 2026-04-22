from core.data_fetcher import DataFetcher
import pandas as pd

def main():
    fetcher = DataFetcher()

    print("Testing data pipeline...")

    df = fetcher.get_latest_data("BTC-USD", limit=10)
    print(f"Latest 10 candls for BTC-USD:")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])

    print(f"\nTotal rows in database: {len(df)}")

if __name__ == "__main__":
    main()