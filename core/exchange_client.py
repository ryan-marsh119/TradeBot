import ccxt
import os
from config.settings import EXCHANGE

def get_exchange():
    if EXCHANGE == "coinbase":
        return ccxt.coinbase({
            'apiKey': os.getenv('COINBASE_API_KEY'),
            'secret': os.getenv('COINBASE_API_SECRET'),
            'enableRateLimit': True,
        }).fetch_accounts()
    elif EXCHANGE == "kraken":
        return ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
        })
    
    raise ValueError(f"Unknown exchange: {EXCHANGE}")