#########################################################################################
#                             const.py                                                  #
# Description: This module contains various constants and enumerations used for         #
# financial data analysis, including ticker symbols, time spans for analysis, and       #
# mappings for different data sources.                                                  #
#                                                                                       #
# Author: DING Yangyang                                                                 #
# Created: 2024-11-24                                                                   #
# Last Modified: 2024-11-28                                                             #
# All rights reserved.                                                                  #
#########################################################################################


import enum

class TimeSpansForAnalysis(enum.Enum):
    SPAN_2001_2003 = "2001-01-03|2003-06-25"
    SPAN_2007_2008 = "2007-09-18|2008-12-16"
    SPAN_2019_2019 = "2019-08-01|2019-10-31"
    SPAN_2020_2020 = "2020-03-03|2020-03-16"
    SPAN_2000_2023 = "2000-01-01|2023-12-31"


class Tickers(enum.Enum):
    SP500 = 'SP500'
    NASDAQ = 'NASDAQ'
    DOWJONES = 'DOWJONES'
    WIND_A = 'WIND_A'
    HS300 = 'HS300'
    CYBI = 'CYBI'
    HSI = 'HSI'
    MSCI_DEVELOPED = 'MSCI_DEVELOPED'
    MSCI_EMERGING = 'MSCI_EMERGING'
    USDI = 'USDI'
    US10Y_TREASURY = 'US10Y_TREASURY'
    CNY10Y_TREASURY = 'CNY10Y_TREASURY'
    GER10Y_TREASURY = 'GER10Y_TREASURY'
    JPN10Y_TREASURY = 'JPN10Y_TREASURY'
    UK10Y_TREASURY = 'UK10Y_TREASURY'
    GOLD = 'GOLD'
    OIL = 'OIL'
    EURUSD = 'EURUSD'
    JPYUSD = 'JPYUSD'
    # GBPUSD = 'GBPUSD'
    CNYUSD = 'CNYUSD'
    FED_FUNDS_RATE = 'FED_FUNDS_RATE'
    # Step 0
    US_INDPRO = 'US_INDPRO'
    US_DGORDER = 'US_DGORDER'
    US_RSAFS = 'US_RSAFS'
    US_PAYEMS = 'US_PAYEMS'
    US_UNRATE = 'US_UNRATE'
    # US_ICSA = 'US_ICSA'
    US_CIVPART = 'US_CIVPART'
    US_CPIAUCSL = 'US_CPIAUCSL'
    US_CPILFESL = 'US_CPILFESL'
    US_PPIACO = 'US_PPIACO'
    US_PCEPI = 'US_PCEPI'
    US_BOPGSTB = 'US_BOPGSTB'
    US_CSCICP03USM665S = 'US_CSCICP03USM665S'
    US_UMCSENT = 'US_UMCSENT'
    US_MTSDS133FMS = 'US_MTSDS133FMS'
    US_HOUST = 'US_HOUST'
    US_CSUSHPISA = 'US_CSUSHPISA'

    

class TushareGlobalIndex(enum.Enum):
    # 美国标普500指数
    SP500 = 'SPX'
    # 纳斯达克指数
    NASDAQ = 'IXIC'
    # 道琼斯指数
    DOWJONES = 'DJI'


class TushareIndex(enum.Enum):
    # 沪深300指数
    HS300 = '000300.SH'
    # 创业板指数
    CYBI = '399006.SZ'


class yFinanceTickers(enum.Enum):
    # 美国标普500指数
    SP500 = '^GSPC'
    # 纳斯达克指数
    NASDAQ = '^IXIC'
    # 道琼斯指数
    DOWJONES = '^DJI'
    # 恒生指数
    HSI = '^HSI'
    # 美国十年期国债收益率
    US10Y = '^TNX'
    # 美元指数
    USDI = 'DX-Y.NYB'
    # 黄金价格
    GOLD = 'GC=F'
    # 石油价格
    OIL = 'CL=F'
    # 欧元兑美元
    EURUSD = 'EURUSD=X'
    # 日元兑美元
    JPYUSD = 'JPYUSD=X'
    # 英镑兑美元
    # GBPUSD = 'GBPUSD=X'
    # RMB兑美元
    CNYUSD = 'CNYUSD=X'
    # MSCI 发达市场指数
    MSCI_DEVELOPED = "IDEV"
    # MSCI 新兴市场指数
    MSCI_EMERGING = "EEM"
    

class FreddieMacSeries(enum.Enum):
    FEDERAL_FUNDS_RATE = 'DFF'
    US10Y_TREASURY = 'DGS10'
    GER10Y_TREASURY = 'IRLTLT01DEM156N'
    JPN10Y_TREASURY = 'IRLTLT01JPM156N'
    UK10Y_TREASURY = 'IRLTLT01GBM156N'
    MORTGAGE_RATE_30Y = 'MORTGAGE30US'
    MORTGAGE_RATE_15Y = 'MORTGAGE15US'
    MORTGAGE_RATE_5Y = 'MORTGAGE5US'
    MORTGAGE_DELINQUENCY_RATE = 'DRSFRMACBS'
    MORTGAGE_APPLICATION = 'MORTGAGE_APP'
    MORTGAGE_PURCHASE_APPLICATION = 'PURCHASE'
    MORTGAGE_REFINANCE_APPLICATION = 'REFINANCE'
    MORTGAGE_REFINANCE_RATE = 'MORTGAGE_REFI'
    MORTGAGE_PURCHASE_RATE = 'MORTGAGE_PURCHASE'
    US_INDPRO = "INDPRO"
    US_DGORDER = "DGORDER"
    US_RSAFS = "RSAFS"
    US_PAYEMS = "PAYEMS"
    US_UNRATE = "UNRATE"
    # US_ICSA = "ICSA"
    US_CIVPART = "CIVPART"
    US_CPIAUCSL = "CPIAUCSL"
    US_CPILFESL = "CPILFESL"
    US_PPIACO = "PPIACO"
    US_PCEPI = "PCEPI"
    US_BOPGSTB = "BOPGSTB"
    US_CSCICP03USM665S = "CSCICP03US"
    US_UMCSENT = "UMCSENT"
    US_MTSDS133FMS = "MTSDS133FMS"
    US_HOUST = "HOUST"
    US_CSUSHPISA = "CSUSHPISA"


TICKER_SYMBOLS = {
    "SP500": yFinanceTickers.SP500.value,
    "NASDAQ": yFinanceTickers.NASDAQ.value,
    "DOWJONES": yFinanceTickers.DOWJONES.value,
    "WIND_A": "WIND_A",
    "HS300": TushareIndex.HS300.value,
    "CYBI": TushareIndex.CYBI.value,
    "HSI": yFinanceTickers.HSI.value,
    "MSCI_DEVELOPED": yFinanceTickers.MSCI_DEVELOPED.value,
    "MSCI_EMERGING": yFinanceTickers.MSCI_EMERGING.value,
    "USDI": yFinanceTickers.USDI.value,
    "US10Y_TREASURY": FreddieMacSeries.US10Y_TREASURY.value,
    "CNY10Y_TREASURY": "CNY10Y_TREASURY",
    "GER10Y_TREASURY": FreddieMacSeries.GER10Y_TREASURY.value,
    "JPN10Y_TREASURY": FreddieMacSeries.JPN10Y_TREASURY.value,
    "UK10Y_TREASURY": FreddieMacSeries.UK10Y_TREASURY.value,
    "GOLD": yFinanceTickers.GOLD.value,
    "OIL": yFinanceTickers.OIL.value,
    "EURUSD": yFinanceTickers.EURUSD.value,
    "JPYUSD": yFinanceTickers.JPYUSD.value,
    # "GBPUSD": yFinanceTickers.GBPUSD.value,
    "CNYUSD": yFinanceTickers.CNYUSD.value,
    "FED_FUNDS_RATE": FreddieMacSeries.FEDERAL_FUNDS_RATE.value,
    "US_INDPRO": FreddieMacSeries.US_INDPRO.value,
    "US_DGORDER": FreddieMacSeries.US_DGORDER.value,
    "US_RSAFS": FreddieMacSeries.US_RSAFS.value,
    "US_PAYEMS": FreddieMacSeries.US_PAYEMS.value,
    "US_UNRATE": FreddieMacSeries.US_UNRATE.value,
    # "US_ICSA": FreddieMacSeries.US_ICSA.value,
    "US_CIVPART": FreddieMacSeries.US_CIVPART.value,
    "US_CPIAUCSL": FreddieMacSeries.US_CPIAUCSL.value,
    "US_CPILFESL": FreddieMacSeries.US_CPILFESL.value,
    "US_PPIACO": FreddieMacSeries.US_PPIACO.value,
    "US_PCEPI": FreddieMacSeries.US_PCEPI.value,
    "US_BOPGSTB": FreddieMacSeries.US_BOPGSTB.value,
    "US_CSCICP03USM665S": FreddieMacSeries.US_CSCICP03USM665S.value,
    "US_UMCSENT": FreddieMacSeries.US_UMCSENT.value,
    "US_MTSDS133FMS": FreddieMacSeries.US_MTSDS133FMS.value,
    "US_HOUST": FreddieMacSeries.US_HOUST.value,
    "US_CSUSHPISA": FreddieMacSeries.US_CSUSHPISA.value,
}

TICKER_SOURCE = {
    "SP500": "yfinance",
    "NASDAQ": "yfinance",
    "DOWJONES": "yfinance",
    "WIND_A": "local",
    "HS300": "local",
    "CYBI": "tushare",
    "HSI": "yfinance",
    "MSCI_DEVELOPED": "local",
    "MSCI_EMERGING": "local",
    "USDI": "yfinance",
    "US10Y_TREASURY": "fred",
    "CNY10Y_TREASURY": "akshare",
    "GER10Y_TREASURY": "fred",
    "JPN10Y_TREASURY": "fred",
    "UK10Y_TREASURY": "fred",
    "GOLD": "local", # Wolrd Gold Council
    "OIL": "yfinance",
    "EURUSD": "local",
    "JPYUSD": "yfinance",
    # "GBPUSD": "yfinance",
    "CNYUSD": "yfinance",
    "FED_FUNDS_RATE": "fred",
    "US_INDPRO": "fred",
    "US_DGORDER": "fred",
    "US_RSAFS": "fred",
    "US_PAYEMS": "fred",
    "US_UNRATE": "fred",
    # "US_ICSA": "fred",
    "US_CIVPART": "fred",
    "US_CPIAUCSL": "fred",
    "US_CPILFESL": "fred",
    "US_PPIACO": "fred",
    "US_PCEPI": "fred",
    "US_BOPGSTB": "fred",
    "US_CSCICP03USM665S": "fred",
    "US_UMCSENT": "fred",
    "US_MTSDS133FMS": "fred",
    "US_HOUST": "fred",
    "US_CSUSHPISA": "fred",
}

TICKER_CHINESE = {
    "SP500": "标普500",
    "NASDAQ": "纳斯达克",
    "DOWJONES": "道琼斯",
    "WIND_A": "万得全A",
    "HS300": "沪深300",
    "CYBI": "创业板",
    "HSI": "恒生指数",
    "MSCI_DEVELOPED": "MSCI发达市场指数",
    "MSCI_EMERGING": "MSCI新兴市场指数",
    "USDI": "美元指数",
    "US10Y_TREASURY": "美国十年期国债收益率",
    "CNY10Y_TREASURY": "中国十年期国债收益率",
    "GER10Y_TREASURY": "德国十年期国债收益率",
    "JPN10Y_TREASURY": "日本十年期国债收益率",
    "UK10Y_TREASURY": "英国十年期国债收益率",
    "GOLD": "黄金价格",
    "OIL": "石油价格",
    "EURUSD": "欧元兑美元",
    "JPYUSD": "日元兑美元",
    # "GBPUSD": "英镑兑美元",
    "CNYUSD": "人民币兑美元",
    "FED_FUNDS_RATE": "联邦基金利率",
    "US_INDPRO": "美国工业生产指数",
    "US_DGORDER": "美国耐用品订单",
    "US_RSAFS": "美国零售销售额",
    "US_PAYEMS": "美国非农就业人数",
    "US_UNRATE": "美国失业率",
    # "US_ICSA": "美国初请失业金人数",
    "US_CIVPART": "美国劳动参与率",
    "US_CPIAUCSL": "美国消费者价格指数(CPI)",
    "US_CPILFESL": "美国核心消费者价格指数(不含食品和能源CPI)",
    "US_PPIACO": "美国生产者价格指数(PPI)",
    "US_PCEPI": "美国个人消费支出价格指数(PCEPI)",
    "US_BOPGSTB": "美国贸易收支",
    "US_CSCICP03USM665S": "美国消费者信心指数（会议委员会）",
    "US_UMCSENT": "美国密歇根大学消费者信心指数（调查）",
    "US_MTSDS133FMS": "美国联邦预算赤字或盈余",
    "US_HOUST": "美国新屋开工数",
    "US_CSUSHPISA": "美国房价指数",
}

TICKER_ENGLISH = {
    "SP500": "S&P500",
    "NASDAQ": "NASDAQ",
    "DOWJONES": "DOWJONES",
    "WIND_A": "WIND A",
    "HS300": "HS300 Index",
    "CYBI": "ChiNext Index",
    "HSI": "Hang Seng Index",
    "MSCI_DEVELOPED": "MSCI Developed Markets Index",
    "MSCI_EMERGING": "MSCI Emerging Markets Index",
    "USDI": "US Dollar Index",
    "US10Y_TREASURY": "American 10-Year Treasury Yield",
    "CNY10Y_TREASURY": "Chinese 10-Year Treasury Yield",
    "GER10Y_TREASURY": "German 10-Year Treasury Yield",
    "JPN10Y_TREASURY": "Japanese 10-Year Treasury Yield",
    "UK10Y_TREASURY": "British 10-Year Treasury Yield",
    "GOLD": "Gold Price",
    "OIL": "Oil Price",
    "EURUSD": "Euro to US Dollar",
    "JPYUSD": "Japanese Yen to US Dollar",
    # "GBPUSD": "British Pound to US Dollar",
    "CNYUSD": "Chinese Yuan to US Dollar",
    "FED_FUNDS_RATE": "Federal Funds Rate",
    "US_INDPRO": "US Industrial Production",
    "US_DGORDER": "US Durable Goods Orders",
    "US_RSAFS": "US Retail Sales",
    "US_PAYEMS": "US Nonfarm Payrolls",
    "US_UNRATE": "US Unemployment Rate",
    # "US_ICSA": "US Initial Claims",
    "US_CIVPART": "US Civilian Labor Force Participation Rate",
    "US_CPIAUCSL": "US Consumer Price Index (CPI)",
    "US_CPILFESL": "US Core Consumer Price Index (CPI)",
    "US_PPIACO": "US Producer Price Index (PPI)",
    "US_PCEPI": "US Personal Consumption Expenditures Price Index (PCEPI)",
    "US_BOPGSTB": "US Balance of Payments",
    "US_CSCICP03USM665S": "US Conference Board Consumer Confidence Index",
    "US_UMCSENT": "US University of Michigan Consumer Sentiment Index",
    "US_MTSDS133FMS": "US Federal Budget Deficit/Surplus",
    "US_HOUST": "US Housing Starts",
    "US_CSUSHPISA": "US Case-Shiller Home Price Index",
}

ENGLISH_TICKER = {v: k for k, v in TICKER_ENGLISH.items()}