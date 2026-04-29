import ccxt
import os
from config.settings import EXCHANGE

def get_exchange():
    """Build and return a configured CCXT exchange client.

    Returns:
        ccxt.Exchange: Authenticated exchange client for the configured provider.

    Raises:
        ValueError: If ``EXCHANGE`` is not one of the supported adapters.

    Notes:
        Reads API credentials from environment variables at call time.
    """
    if EXCHANGE == "coinbase":
        exchange = ccxt.coinbase({
            'apiKey': os.getenv('COINBASE_API_KEY'),
            'secret': os.getenv('COINBASE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        return exchange
    elif EXCHANGE == "kraken":
        exchange =ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY'),
            'secret': os.getenv('KRAKEN_API_SECRET'),
            'enableRateLimit': True,
        })
        return exchange
    
    raise ValueError(f"Unknown exchange: {EXCHANGE}")