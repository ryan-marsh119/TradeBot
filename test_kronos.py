import sys
import pandas as pd

sys.path.append("./Kronos")

from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")

predictor = KronosPredictor(model, tokenizer, max_context=512)

print("Kronos loaded successfully!")

from coinbase import jwt_generator
from config.settings import API_KEY, API_SECRET

print("Testing coinbase api")

request_method = "GET"
request_path = "/api/v3/brokerage/accounts"

jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
jwt_token = jwt_generator.build_rest_jwt(jwt_uri, API_KEY, API_SECRET)

print(jwt_token)

# from core.exchange_client import get_exchange
# exchange = get_exchange()
print("Connected successfully!")

# print("SOL/USDC Ticker: ", exchange.fetch_ticker('SOL/USDC'))
# print("Balance: ", exchange.fetch_balance())