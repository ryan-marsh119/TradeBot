from core.data_fetcher import DataFetcher

def main():
    print("Starting historical data backfill...")
    fetcher = DataFetcher()
    fetcher.backfill_all()
    print("\nBackfill completed!")

if __name__ == "__main__":
    main()