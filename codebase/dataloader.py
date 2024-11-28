#########################################################################################
#                             dataloader.py                                             #
# Description: This module contains the FinanceDataLoader class, which is used to       #
# load financial data from various sources such as yfinance, tushare, Fred, and local.  #
# It includes methods to load data, cache data, and retrieve data from different APIs.  #
#                                                                                       #
# Author: DING Yangyang                                                                 #
# Created: 2024-11-24                                                                   #
# Last Modified: 2024-11-28                                                             #
# All rights reserved.                                                                  #
#########################################################################################






import os
import datetime
import yfinance as yf
import pandas as pd
import tushare as ts
import numpy as np
import pandas as pd
from fredapi import Fred
from loguru import logger
from codebase.const import *


class FinanceDataLoader:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.fred = Fred(api_key="your_fred_api_key")
        self.pro = ts.pro_api('your_tushare_api_key')

    def load_or_get_data(self, ticker_name: str, source: str = "yfinance", enable_cache: bool = True) -> pd.DataFrame:
        """
        使用 yfinance 或 tushare 获取指定金融指数在给定时间段内的历史数据。

        :param ticker_symbol: 金融指数的 yfinance ticker symbol
        :param start_date: 开始日期 (格式：'YYYY-MM-DD')
        :param end_date: 结束日期 (格式：'YYYY-MM-DD')
        :return: 包含日期、开盘价、收盘价、最高价、最低价、成交量的 pandas DataFrame
        """
        os.makedirs("data", exist_ok=True)
        cache_dir = os.path.join("data", f"{self.start_date}_{self.end_date}")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{TICKER_CHINESE[ticker_name]}.csv")
        try:
            if os.path.exists(cache_path) and enable_cache:
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from cache: {cache_path}")
                return pd.read_csv(cache_path)            
            
            if source == "yfinance":
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from yfinance.")
                df = self.get_yfinance_data(ticker_name)
            elif source == "tushare":
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from Tushare.")
                df = self.get_tushare_data(ticker_name)
            elif source == "fred":
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from Fred.")
                df = self.get_freddie_mac_data(ticker_name)
            elif source == "local":
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from local.")
                df = self.get_local_data(ticker_name)
            elif ticker_name == "CNY10Y_TREASURY" and source == "akshare":
                logger.info(f"Loading [{TICKER_CHINESE[ticker_name]}] data from akshare.")
                df = self.get_cn_bond_data()
            else:
                raise ValueError("Invalid source. Please choose from 'yfinance', 'tushare', or 'fred'.")
        except Exception as e:
            logger.error(f"Failed to load data for [{TICKER_CHINESE[ticker_name]}]. Error: {e}")
            return None
        
        if df is None:
            return None
        
        with open(cache_path, "w") as f:
            df.to_csv(f, index=False)
        return df
    
    def get_tushare_data(self, ticker_name: str) -> pd.DataFrame:
        """
        使用 tushare 获取指定金融指数在给定时间段内的历史数据。

        :param ticker_name: 金融指数的名称
        :param start_date: 开始日期 (格式：'YYYYMMDD')
        :param end_date: 结束日期 (格式：'YYYYMMDD')
        :return: 包含日期、开盘价、收盘价、最高价、最低价、成交量的 pandas DataFrame
        """
        ticker_symbol = TICKER_SYMBOLS[ticker_name]
        start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y%m%d")
        # 获取金融数据
        if ticker_symbol in [m.value for m in TushareIndex]:
            stock_data = self.pro.index_daily(ts_code=ticker_symbol, start_date=start_date, end_date=end_date)
        elif ticker_symbol in [m.value for m in TushareGlobalIndex]:
            stock_data = self.pro.index_global(ts_code=ticker_symbol, start_date=start_date, end_date=end_date)
        else:
            stock_data = self.pro.daily(ts_code=ticker_symbol, start_date=start_date, end_date=end_date)
        # 选取所需的列
        if stock_data.empty:
            return None
        
        stock_data = stock_data[['trade_date', 'close']]
        
        # 重命名列名称
        stock_data.columns = ['Date', 'Data']
        
        # tushare 返回的日期格式为 'yyyymmdd'，需要转换为 'yyyy-mm-dd' 格式
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y%m%d')
        
        return stock_data
    
    def get_yfinance_data(self, ticker_name: str) -> pd.DataFrame:
        """
        使用 yfinance 获取指定金融指数在给定时间段内的历史数据。

        :param ticker_name: 金融指数的名称
        :param start_date: 开始日期 (格式：'YYYY-MM-DD')
        :param end_date: 结束日期 (格式：'YYYY-MM-DD')
        :return: 包含日期、开盘价、收盘价、最高价、最低价、成交量的 pandas DataFrame
        """
        ticker_symbol = TICKER_SYMBOLS[ticker_name]
        # end date should be the next day of the end date
        end_date = (datetime.datetime.strptime(self.end_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        # 获取金融数据
        stock_data = yf.download(ticker_symbol, start=self.start_date, end=end_date)
        if stock_data.empty:
            return None
        # 选取所需的列并重命名
        stock_data = stock_data[['Adj Close']].reset_index()
        
        # 重命名列名称
        stock_data.columns = ['Date', 'Data']
        
        return stock_data
    
    def get_freddie_mac_data(self, ticker_name: str) -> pd.DataFrame:
        """
        使用 Fred API 获取 Freddie Mac 指定数据系列在给定时间段内的历史数据。

        :param ticker_name: Freddie Mac 数据系列的名称
        :param start_date: 开始日期 (格式：'YYYY-MM-DD')
        :param end_date: 结束日期 (格式：'YYYY-MM-DD')
        :return: 包含日期和数据的 pandas DataFrame
        """
        series_id = TICKER_SYMBOLS[ticker_name]
        
        # 获取金融数据
        # 把 YYYY-MM-DD 格式的日期转换为 datetime 格式
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        try:
            stock_data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        except ValueError:
            return None
        if stock_data.empty:
            return None
        # 重命名列名称
        stock_data = pd.DataFrame(stock_data).reset_index()
        stock_data.columns = ['Date', 'Data']
        
        return stock_data
    
    def get_local_data(self, ticker_name: str) -> pd.DataFrame:
        """
        获取本地数据。

        :return: 包含日期和数据的 pandas DataFrame
        """
        df = pd.read_excel(f"data/{TICKER_CHINESE[ticker_name]}.xlsx")
        df.columns = ["ticker", "name", "date", "open", "high", "low", "close", "pct_change", "amount", "volume"]
        stock_data = df[["date", "close"]].dropna()
        stock_data.columns = ["Date", "Data"]
        # select the data in the given date range
        stock_data = stock_data[(stock_data["Date"] >= self.start_date) & (stock_data["Date"] <= self.end_date)].copy().reset_index(drop=True)
        return stock_data

    def get_cn_bond_data(self) -> pd.DataFrame:
        """
        获取中国国债数据。

        :return: 包含日期和数据的 pandas DataFrame
        """
        import akshare as ak
        earliest_date = "19901219"
        
        if datetime.datetime.strptime(self.start_date, "%Y-%m-%d") < datetime.datetime.strptime(earliest_date, "%Y%m%d"):
            logger.warning(f"Start date is earlier than the earliest date available. Using the earliest date as the start date.")
            start_date = earliest_date
        else:
            start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d").strftime("%Y%m%d")
        bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date=start_date)
        df = bond_zh_us_rate_df[["日期", "中国国债收益率10年"]].copy()
        df.columns = ["Date", "Data"]
        df["Date"] = pd.to_datetime(df["Date"])
        # select the data in the given date range
        df = df[(df["Date"] >= self.start_date) & (df["Date"] <= self.end_date)].copy().reset_index(drop=True)
        return df
        