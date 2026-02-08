import os
import logging
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Date, String, Float, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
Base = declarative_base()


class StockPrice(Base):
    __tablename__ = 'nasdaq_prices'
    date = Column(Date, primary_key=True)
    ticker = Column(String(10), primary_key=True)
    adj_close = Column(Float, nullable=False)


class MarketDataPipeline:
    def __init__(self, db_url, sql_view_path='sql/create_view.sql'):
        self.engine = create_engine(db_url)
        self.sql_view_path = sql_view_path
        self._init_db()

    def _init_db(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DROP VIEW IF EXISTS view_nasdaq_returns CASCADE;"))
                conn.commit()
            Base.metadata.drop_all(self.engine)
            Base.metadata.create_all(self.engine)
            self._create_view()
        except SQLAlchemyError as e:
            logger.error(f"DB Init Failed: {e}")

    def _create_view(self):
        if not os.path.exists(self.sql_view_path):
            self.sql_view_path = os.path.join('..', self.sql_view_path)
        try:
            with open(self.sql_view_path, 'r') as f:
                sql = f.read()
            with self.engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()
        except Exception as e:
            logger.error(f"View Creation Failed: {e}")

    def run(self, ticker, start, end):
        logger.info(f"Fetching {ticker}...")
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            logger.warning(f"No data fetched for ticker {ticker}")
            return

        # Handling MultiIndex complexity of yfinance
        try:
            df = data.xs('Close', level=0, axis=1) if isinstance(data.columns, pd.MultiIndex) else data['Close']
        except KeyError:
            df = data['Close']

        if isinstance(df, pd.Series):
            df = df.to_frame(name=ticker)

        else:
            df.columns = [ticker]

        df = df.stack().reset_index()
        df.columns = ['date', 'ticker', 'adj_close']

        with self.engine.begin() as conn:
            df.to_sql('nasdaq_prices', conn, if_exists='append', index=False, chunksize=1000)
        logger.info(f"Loaded {len(df)} rows.")


if __name__ == "__main__":
    # Ingest 24 years of Nasdaq Data
    pipeline = MarketDataPipeline(DATABASE_URL)
    pipeline.run('^NDX', '2000-01-01', '2024-01-01')