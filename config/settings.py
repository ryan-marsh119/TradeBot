import os
from dotenv import load_dotenv

load_dotenv()

PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
EXCHANGE = os.getenv("EXCHANGE", "coinbase")
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")