from core.exchange_client import get_exchange
from config.settings import EXCHANGE, PAPER_MODE, API_KEY, API_SECRET

def test_connection():
    print(f"Testing connection to {EXCHANGE.upper()}...")
    print(f"Paper mode: {PAPER_MODE}\n")

    try:
        exchange = get_exchange()

        ticker = exchange.fetch_ticker('BTC/USD')
        print(f"BTC/USD Ticker:")
        print(f" Last price: ${ticker['last']:.4f}")
        print(f" Bid price: ${ticker['bid']:.4f}")
        print(f" Ask price: ${ticker['ask']:.4f}")
        print(f" High price: ${ticker['high']}")
        print(f" Low price: ${ticker['low']}")
        print(f" Volume: {ticker['quoteVolume']}")

        balance = exchange.fetch_balance()
        print(f"Balance retrieved successfully!")
        print(f"Total USDC balance: {balance.get('USDC', {}).get('total', 0)} USDC")
        print(f"Available USDC balance: {balance.get('USDC', {}).get('available', 0)} USDC")
        print(f"Total BTC balance: {balance.get('BTC', {}).get('total', 0)} BTC")
        print(f"Available BTC balance: {balance.get('BTC', {}).get('available', 0)} BTC")
        print(f"Total USD balance: {balance.get('USD', {}).get('total', 0)} USD")
        print(f"Available USD balance: {balance.get('USD', {}).get('available', 0)} USD")
        print(f"Total SOL balance: {balance.get('SOL', {}).get('total', 0)} SOL")
        print(f"Available SOL balance: {balance.get('SOL', {}).get('available', 0)} SOL")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()