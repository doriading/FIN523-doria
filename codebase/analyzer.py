#########################################################################################
#                             analyzer.py                                               #
# Description: This module contains the FinancialDataAnalyzer class, which is used to   #
# analyze financial data. It includes methods to load data, calculate statistics,       #
# covariance, correlation, change ratio, select high correlation, and perform regression#
# analysis.                                                                             #
#                                                                                       #
# Author: DING Yangyang                                                                 #
# Created: 2024-11-24                                                                   #
# Last Modified: 2024-11-28                                                             #
# All rights reserved.                                                                  #
#########################################################################################


import os 
import time
from numpy.polynomial.polynomial import Polynomial 
import pandas as pd 
import numpy as np
from loguru import logger
from typing import Dict, Any
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import statsmodels.api as sm
from codebase.const import *
from typing import List
from codebase.dataloader import FinanceDataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR


class FinancialDataAnalyzer:
    def __init__(self, start_date: str, end_date: str): 
        """
        初始化金融数据分析器。
        设置开始日期和结束日期，并初始化数据加载器，统计数据、协方差、相关系数、涨跌幅等属性，以及高相关性金融指数对和回归分析结果。
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_loader = FinanceDataLoader(start_date, end_date)
        self.data: Dict[str, pd.DataFrame] = {}
        self.exog_data: Dict[str, pd.DataFrame] = {}
        self.statistics: pd.DataFrame = pd.DataFrame()
        self.covariance: pd.DataFrame = pd.DataFrame()
        self.correlation: pd.DataFrame = pd.DataFrame()
        self.change_ratio: pd.DataFrame = pd.DataFrame()
        self.high_correlation: Dict[str, Any] = {}
        self.linear_regression_results: Dict[str, Any] = {}
        self.polynomial_regression_results: Dict[str, Any] = {}
        self.exog_ticker_names = [
            Tickers.US_INDPRO.value, Tickers.US_DGORDER.value, Tickers.US_RSAFS.value, Tickers.US_PAYEMS.value,
            Tickers.US_UNRATE.value, Tickers.US_CIVPART.value, Tickers.US_CPIAUCSL.value, Tickers.US_CPILFESL.value,
            Tickers.US_PPIACO.value, Tickers.US_PCEPI.value, Tickers.US_BOPGSTB.value, Tickers.US_CSCICP03USM665S.value, 
            Tickers.US_UMCSENT.value, Tickers.US_MTSDS133FMS.value, Tickers.US_HOUST.value, Tickers.US_CSUSHPISA.value, 
            Tickers.FED_FUNDS_RATE.value,
        ]
    
    def load_data(self, enable_cache: bool = True):
        """
        加载所有金融指数数据。
        """
        for ticker in TICKER_SYMBOLS:
            df = self.data_loader.load_or_get_data(ticker, TICKER_SOURCE[ticker], enable_cache)
            # time.sleep(1)
            if df is not None:
                df.dropna(inplace=True)
                if not df.empty:
                    if ticker in self.exog_ticker_names:
                        self.exog_data[ticker] = df.copy()
                    else:
                        self.data[ticker] = df.copy()
            else:
                logger.debug(f"Failed to load data for {TICKER_ENGLISH[ticker]}.")
        logger.info(f"Loaded {len(self.data)} financial indices.")

    def calculate_statistics(self) -> pd.DataFrame:
        """
        计算金融指数的统计数据，并保存在 pd.DataFrame 中。
        Index 为金融指数的名称，Columns 为统计数据。需要计算的统计数据有：skewness, kurtosis, mean, std, median, quantile(0.25), quantile(0.75).
        """
        all_statistics = {}
        for ticker in self.data:
            data = self.data[ticker]["Data"]
            statistics = {}
            statistics["skewness"] = data.skew()
            statistics["kurtosis"] = data.kurtosis()
            statistics["mean"] = data.mean()
            statistics["std"] = data.std()
            statistics["median"] = data.median()
            statistics["quantile_25"] = data.quantile(0.25)
            statistics["quantile_75"] = data.quantile(0.75)
            all_statistics[ticker] = pd.DataFrame(statistics, index=[TICKER_ENGLISH[ticker]])
        if len(all_statistics) > 0:
            self.statistics = pd.concat(all_statistics.values())
        os.makedirs("output/statistics", exist_ok=True)
        self.change_ratio.to_csv(f"output/statistics/statistics_{self.start_date}_{self.end_date}.csv", index=True)
        return self.statistics

    def calculate_covariance(self) -> pd.DataFrame:
        """
        计算金融指数之间的协方差矩阵。
        """
        data = pd.concat([self.data[ticker]["Data"] for ticker in self.data], axis=1)
        self.covariance = data.cov()
        self.covariance.columns = [TICKER_ENGLISH[ticker] for ticker in self.data]
        self.covariance.index = [TICKER_ENGLISH[ticker] for ticker in self.data]
        os.makedirs("output/covariance", exist_ok=True)
        self.change_ratio.to_csv(f"output/covariance/covariance_{self.start_date}_{self.end_date}.csv", index=True)
        return self.covariance
    
    def calculate_correlation(self) -> pd.DataFrame:
        """
        计算金融指数之间的相关系数矩阵。
        """
        data = pd.concat([self.data[ticker]["Data"] for ticker in self.data], axis=1)
        self.correlation = data.corr()
        self.correlation.columns = [TICKER_ENGLISH[ticker] for ticker in self.data]
        self.correlation.index = [TICKER_ENGLISH[ticker] for ticker in self.data]
        os.makedirs("output/correlation", exist_ok=True)
        self.correlation.to_csv(f"output/correlation/correlation_{self.start_date}_{self.end_date}.csv", index=True)
        return self.correlation
    
    def calculate_change_ratio(self) -> pd.DataFrame:
        """
        计算金融指数的涨跌幅。
        """
        all_change_ratio = {}
        for key in self.data:
            if key in [
                Tickers.FED_FUNDS_RATE.value, Tickers.US10Y_TREASURY.value, Tickers.GER10Y_TREASURY.value,
                Tickers.JPN10Y_TREASURY.value, Tickers.UK10Y_TREASURY.value, Tickers.CNY10Y_TREASURY.value
            ]:
                change_ratio = self.data[key]["Data"].diff().cumsum().iloc[-1]
                all_change_ratio[TICKER_ENGLISH[key]] = change_ratio
                # logger.debug(f"Change ratio for {TICKER_ENGLISH[key]}: {change_ratio}")
            else:
                change_ratio = self.data[key]["Data"].diff().cumsum().iloc[-1] / self.data[key]["Data"].iloc[0] * 100
                all_change_ratio[TICKER_ENGLISH[key] + "(%)"] = change_ratio
            
        self.change_ratio = pd.DataFrame(all_change_ratio, index=[f"{self.start_date} - {self.end_date}"]).T
        os.makedirs("output/change_ratio", exist_ok=True)
        self.change_ratio.to_csv(f"output/change_ratio/change_ratio_{self.start_date}_{self.end_date}.csv", index=True)
        return self.change_ratio
    
    
    def select_high_correlation(self, threshold: float = 0.8):
        """
        选择相关系数大于阈值的金融指数对。
        """
        high_correlation = {}
        exog_ticker_names = [
            Tickers.US_INDPRO.value, Tickers.US_DGORDER.value, Tickers.US_RSAFS.value, Tickers.US_PAYEMS.value,
            Tickers.US_UNRATE.value, Tickers.US_CIVPART.value, Tickers.US_CPIAUCSL.value, Tickers.US_CPILFESL.value,
            Tickers.US_PPIACO.value, Tickers.US_PCEPI.value, Tickers.US_BOPGSTB.value, Tickers.US_CSCICP03USM665S.value, Tickers.US_UMCSENT.value,
            Tickers.US_MTSDS133FMS.value, Tickers.US_HOUST.value, Tickers.US_CSUSHPISA.value, Tickers.FED_FUNDS_RATE.value,
            ]
        for ticker1 in self.correlation:
            for ticker2 in self.correlation:
                if ticker1 == ticker2:
                    continue
                if ticker1 + "|" + ticker2 in high_correlation or ticker2 + "|" + ticker1 in high_correlation:
                    continue
                if abs(self.correlation[ticker1][ticker2]) >= threshold:
                    high_correlation[ticker1 + "|" + ticker2] = self.correlation[ticker1][ticker2]
        self.high_correlation = pd.DataFrame(high_correlation, index=["Correlation"]).T
        os.makedirs("output/high_correlation", exist_ok=True)
        self.high_correlation.to_csv(f"output/high_correlation/high_correlation_{self.start_date}_{self.end_date}.csv", index=True)
        return self.high_correlation
    
    # def regression_analysis(self) -> pd.DataFrame:
    #     """
    #     对 correlation 大于阈值的金融指数对进行回归分析。
    #     选择线性回归和多项式回归模型进行分析。
    #     检验模型的显著性，包括 R^2、p-value、t-value、F-value 和 F-statistic。
    #     保存 R^2 最大的模型类型和结果。
    #     """
    #     r2_results = {}
    #     p_value_results = {}
    #     t_value_results = {}
    #     f_value_results = {}
    #     f_statistic_results = {}
    #     for ticker_pair in self.high_correlation.index.to_list():
    #         ticker1, ticker2 = ticker_pair.split("|")
    #         ticker1_name = ENGLISH_TICKER[ticker1]
    #         ticker2_name = ENGLISH_TICKER[ticker2]
    #         linear_regression = self.linear_regression(ticker1_name, ticker2_name)
    #         polynormial_regression = self.polynomial_regression(ticker1_name, ticker2_name)
            
    #         r2_results[ticker_pair] = {
    #             "Linear Regression": linear_regression["R^2"].values[0],
    #             "Polynomial Regression": polynormial_regression["R^2"].values[0],
    #         }
    #         p_value_results[ticker_pair] = {
    #             "Linear Regression": linear_regression["p-value"].values[0],
    #         }
    #         t_value_results[ticker_pair] = {
    #             "Linear Regression": linear_regression["t-value"].values[0],
    #         }
    #         f_value_results[ticker_pair] = {
    #             "Polynomial Regression": polynormial_regression["F-pvalue"].values[0],
    #         }
    #         f_statistic_results[ticker_pair] = {
    #             "Polynomial Regression": polynormial_regression["F-statistic"].values[0],
    #         }
            
    #         max_r2 = max(r2_results[ticker_pair].values())
    #         max_model = [k for k, v in r2_results[ticker_pair].items() if v == max_r2][0]
    #         self.regression_results[ticker_pair] = {
    #             "Model": max_model,
    #             "R^2": max_r2,
    #             "p-value": p_value_results[ticker_pair].get(max_model, None),
    #             "t-value": t_value_results[ticker_pair].get(max_model, None),
    #             "f-value": f_value_results[ticker_pair].get(max_model, None),
    #             "f-statistic": f_statistic_results[ticker_pair].get(max_model, None),
    #         }
    #     return pd.DataFrame(self.regression_results).T      

    def overall_linear_regression_analysis(self, plot: bool = False) -> pd.DataFrame:
        """
        对 correlation 大于阈值的金融指数对进行线性回归分析。
        检验模型的显著性，包括 R^2、p-value、t-value。
        """
        for ticker_pair in self.high_correlation.index.to_list():
            ticker1, ticker2 = ticker_pair.split("|")
            ticker1_name = ENGLISH_TICKER[ticker1]
            ticker2_name = ENGLISH_TICKER[ticker2]
            linear_regression = self.linear_regression(ticker1_name, ticker2_name, plot=plot)
            if linear_regression is None:
                continue

            self.linear_regression_results[ticker_pair] = {
                "Model": "Linear Regression",
                "R^2": linear_regression["R^2"].values[0],
                "p-value": linear_regression["p-value"].values[0],
                "t-value": linear_regression["t-value"].values[0],
            }
        self.linear_regression_results = pd.DataFrame(self.linear_regression_results).T
        os.makedirs("output/linear_regression", exist_ok=True)
        self.linear_regression_results.to_csv(f"output/linear_regression/linear_regression_{self.start_date}_{self.end_date}.csv", index=True)
        return self.linear_regression_results

    def overall_polynomial_regression_analysis(self, plot: bool = False) -> pd.DataFrame:
        """
        对 correlation 大于阈值的金融指数对进行多项式回归分析。
        检验模型的显著性，包括 R^2、F-value 和 F-statistic。
        """
        for ticker_pair in self.high_correlation.index.to_list():
            ticker1, ticker2 = ticker_pair.split("|")
            ticker1_name = ENGLISH_TICKER[ticker1]
            ticker2_name = ENGLISH_TICKER[ticker2]
            polynormial_regression = self.polynomial_regression(ticker1_name, ticker2_name, plot=plot)
            if polynormial_regression is None:
                continue
            
            self.polynomial_regression_results[ticker_pair] = {
                "Model": "Polynomial Regression",
                "R^2": polynormial_regression["R^2"].values[0],
                "f-value": polynormial_regression["F-pvalue"].values[0],
                "f_statistic": polynormial_regression["F-statistic"].values[0]
            }
        self.polynomial_regression_results = pd.DataFrame(self.polynomial_regression_results).T
        os.makedirs("output/polynomial_regression", exist_ok=True)
        self.polynomial_regression_results.to_csv(f"output/polynomial_regression/polynomial_regression_{self.start_date}_{self.end_date}.csv", index=True)
        return self.polynomial_regression_results
    
    def overall_linear_regression_multi_factor_analysis(self, plot: bool = False) -> pd.DataFrame:
        """
        对 correlation 大于阈值的金融指数对进行多因子线性回归分析。
        检验模型的显著性，包括 R^2、p-value、t-value。
        """
        exog_ticker_names = [
            Tickers.US_INDPRO.value, Tickers.US_DGORDER.value, Tickers.US_RSAFS.value, Tickers.US_PAYEMS.value,
            Tickers.US_UNRATE.value, Tickers.US_CIVPART.value, Tickers.US_CPIAUCSL.value, Tickers.US_CPILFESL.value,
            Tickers.US_PPIACO.value, Tickers.US_PCEPI.value, Tickers.US_BOPGSTB.value, Tickers.US_CSCICP03USM665S.value, Tickers.US_UMCSENT.value,
            Tickers.US_MTSDS133FMS.value, Tickers.US_HOUST.value, Tickers.US_CSUSHPISA.value, Tickers.FED_FUNDS_RATE.value,
            ]
        regression_result = {}
        for ticker_name in self.data:
            if ticker_name not in exog_ticker_names:
                result = self.linear_regression_multi_factors(ticker_name, exog_ticker_names, plot=plot)
                if result is not None:
                    regression_result[TICKER_ENGLISH[ticker_name]] = result
        regression_df = pd.concat(regression_result.values(), keys=regression_result.keys())
        os.makedirs("output/linear_regression_multi_factors", exist_ok=True)
        regression_df.to_csv(f"output/linear_regression_multi_factors/linear_regression_multi_factors_{self.start_date}_{self.end_date}.csv", index=True)
        return regression_df


    def linear_regression(self, ticker1_name: str, ticker2_name: str, plot: bool = False) -> pd.DataFrame:
        """
        使用线性回归模型分析两个金融指数之间的关系，并可视化结果。
        :param ticker1_name: 第一个金融指数的名称（作为自变量 Data_1）
        :param ticker2_name: 第二个金融指数的名称（作为目标变量 Data_2）
        :return: 包含线性回归模型的系数、截距、R^2、均方误差 (MSE)、t 值和 p 值的 pandas DataFrame
        """
        # 从数据集中获取两个金融指数的时间序列
        df1 = self.data[ticker1_name]
        df2 = self.data[ticker2_name]
        
        # 合并两个金融指数的历史数据，并过滤异常值
        df = df1.merge(df2, on='Date', suffixes=('_1', '_2'), how='inner')
        # df = df[(df['Data_1'].between(-1e-10, 50000)) & (df['Data_2'].between(-1e-10, 50000))]
        if df.shape[0] < 3:
            return None
        
        # 自变量（X）和目标变量（Y）
        X = df[['Data_1']].values  # 转为 numpy 格式的一维数组
        Y = df['Data_2'].values  # 转为 numpy 格式的一维数组
        
        # 初始化线性回归模型
        model = LinearRegression()
        
        # 使用线性回归模型拟合数据
        model.fit(X, Y)
        
        # 获取模型的参数（系数和截距）
        coefficient = model.coef_[0]  # 回归系数
        intercept = model.intercept_  # 截距
        
        # 预测目标变量
        Y_pred = model.predict(X)
        
        # 计算模型性能指标
        mse = mean_squared_error(Y, Y_pred)  # 均方误差
        r2 = r2_score(Y, Y_pred)  # 拟合优度 R^2
        
        # 计算 t 值和 p 值
        n = len(X)  # 样本数量
        se = np.sqrt(np.sum((Y - Y_pred) ** 2) / (n - 2)) / np.sqrt(np.sum((X - np.mean(X)) ** 2))  # 标准误差
        t_value = coefficient / se  # t 值
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=n - 2))  # p 值
        
        # 可视化结果
        if plot:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.scatter(X, Y, color='blue', alpha=0.6, label='Actual Data')
            plt.plot(X, Y_pred, color='red', label=f'Linear Fit: y = {intercept:.2f} + {coefficient:.2f}x')
            plt.xlabel(ticker1_name)
            plt.ylabel(ticker2_name)
            plt.title(f'Linear Regression: R^2 = {r2:.2f}, MSE = {mse:.2f}')
            plt.legend()
            plt.grid()
            plt.show()
            
        # 构建结果 DataFrame
        results = {
            'Coefficient (Slope)': [coefficient],
            'Intercept': [intercept],
            'R^2': [r2],
            'MSE': [mse],
            't-value': [t_value],
            'p-value': [p_value]
        }
        
        return pd.DataFrame(results)
    
    def linear_regression_multi_factors(self, endog_ticker_name: str, exog_ticker_names: List[str], plot: bool = False) -> pd.DataFrame:
        """
        使用线性回归模型分析一个金融指数与多个金融指数之间的关系，并可视化结果。
        :param endog_ticker_name: 作为因变量的金融指数名称
        :param exog_ticker_names: 作为自变量的金融指数名称列表
        :param plot: 是否绘制回归结果图
        :return: 包含线性回归模型的系数、截距、R^2、均方误差 (MSE)、t 值和 p 值的 pandas DataFrame
        """
        # 从数据集中获取因变量和自变量的时间序列
        df_endog = self.data[endog_ticker_name]
        # 判断 df_endog 是否是日频
        # if pd.infer_freq(df_endog.index) == 'D':
        #     # 生成完整的日期范围
        #     full_date_range = pd.date_range(start=df_endog.index.min(), end=df_endog.index.max(), freq='D')
        #     # 重新索引并使用线性插值法补全数据
        #     df_endog = df_endog.reindex(full_date_range).interpolate(method='linear')
        existing_exog_ticker_names = [ticker for ticker in exog_ticker_names if ticker in self.exog_data]
        df_exog = [self.exog_data[ticker] for ticker in exog_ticker_names if ticker in self.exog_data]
        
        # 合并所有金融指数的历史数据
        df = df_endog
        for i, df_ex in enumerate(df_exog):
            df = df.merge(df_ex, on='Date', suffixes=('', f'_{i}'), how='inner')
        
        if df.shape[0] < 3:
            logger.debug(f"Insufficient data for {endog_ticker_name} and {exog_ticker_names}.")
            print(df)
            return None
        
        # 因变量（Y）和自变量（X）
        Y = df['Data'].values  # 因变量
        X = df[[f'Data_{i}' for i in range(len(df_exog))]].values  # 自变量
        
        # 添加截距项（constant）并构建 statsmodels 的回归模型
        X_with_intercept = sm.add_constant(X)  # 增加截距列
        model = sm.OLS(Y, X_with_intercept).fit()  # 用最小二乘法拟合
        
        # 获取模型的参数（系数和截距）
        coefficients = model.params  # 回归系数和截距
        intercept = coefficients[0]  # 截距
        slopes = coefficients[1:]  # 回归系数
        
        # 预测目标变量
        Y_pred = model.predict(X_with_intercept)
        
        # 计算模型性能指标
        mse = mean_squared_error(Y, Y_pred)  # 均方误差
        r2 = r2_score(Y, Y_pred)  # 拟合优度 R^2
        
        # 获取 t 值和 p 值
        t_values = model.tvalues
        p_values = model.pvalues
        
        # 可视化结果
        if plot:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.scatter(Y, Y_pred, color='blue', alpha=0.6, label='Actual vs Predicted')
            plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', label='Perfect Fit')
            plt.xlabel(f'Actual {TICKER_ENGLISH[endog_ticker_name]}')
            plt.ylabel(f'Predicted {TICKER_ENGLISH[endog_ticker_name]}')
            plt.title(f'Linear Regression Multi Factors: {TICKER_ENGLISH[endog_ticker_name]} R^2 = {r2:.2f}, MSE = {mse:.2f}')
            plt.legend()
            plt.grid()
            plt.show()
        
        # 构建结果 DataFrame
        results = {
            'Intercept': [intercept],
            'R^2': [r2],
            'MSE': [mse]
        }
        for ticker in exog_ticker_names:
            if ticker in self.exog_data:
                i = existing_exog_ticker_names.index(ticker)
                results[f'Coefficient (Slope) {ticker}'] = [slopes[i]]
                results[f't-value {ticker}'] = [t_values[i+1]]
                results[f'p-value {ticker}'] = [p_values[i+1]]
            else:
                results[f'Coefficient (Slope) {ticker}'] = [0]
                results[f't-value {ticker}'] = [0]
                results[f'p-value {ticker}'] = [0]
        
        return pd.DataFrame(results)
       
    def polynomial_regression(self, ticker1_name: str, ticker2_name: str, degree: int = 2, plot: bool = False) -> pd.DataFrame:
        """
        使用多项式回归分析两个金融指数之间的关系。
        :param ticker1_name: 第一个金融指数的名称（作为自变量 Data_1）
        :param ticker2_name: 第二个金融指数的名称（作为目标变量 Data_2）
        :param degree: 多项式的阶数，默认为 2
        :return: 包含回归系数、R^2 和均方误差的 pandas DataFrame
        """
        # 从数据集中获取两个金融指数的时间序列
        df1 = self.data[ticker1_name]
        df2 = self.data[ticker2_name]
        
        # 合并两个金融指数的历史数据
        df = df1.merge(df2, on='Date', suffixes=('_1', '_2'), how='inner')
        df = df[(df['Data_1'].between(0, 50000)) & (df['Data_2'].between(0, 50000))]
        if df.shape[0] < 8:
            return None
        
        # 自变量（X）和目标变量（Y）
        X = df[['Data_1']].values  # 自变量，需要是二维数组
        Y = df['Data_2'].values  # 目标变量
        
        # 对自变量进行多项式特征扩展
        X = np.array(X).reshape(-1, 1)  # 转为二维
        Y = np.array(Y)
        
        # Step 1: 多项式特征扩展
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Step 2: 添加截距项（constant）并构建 statsmodels 的回归模型
        X_poly_with_intercept = sm.add_constant(X_poly)  # 增加截距列
        model = sm.OLS(Y, X_poly_with_intercept).fit()   # 用最小二乘法拟合
        
        # Step 3: 获取统计指标
        # F-statistic 和对应的 p-value（全局显著性检验）
        f_statistic = model.fvalue      # F 统计量
        f_pvalue = model.f_pvalue       # F 检验对应的 p 值
        
        # R² 和调整后的 R²
        r2 = model.rsquared             # R²
        adj_r2 = model.rsquared_adj     # 调整后的 R²
        
        # AIC 和 BIC
        aic = model.aic                 # AIC
        bic = model.bic                 # BIC
        
        # Step 4: 打印回归模型摘要
        print(model.summary())          # 打印详细模型结果
        
        # 可视化实际值、预测值以及不确定性
        if plot:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.scatter(X, Y, color='blue', label='Actual Data', alpha=0.6)
            plt.plot(X, model.fittedvalues, color='red', label='Polynomial Regression Prediction')
            plt.xlabel(TICKER_ENGLISH[ticker1_name])
            plt.ylabel(TICKER_ENGLISH[ticker2_name])
            plt.title(f'Polynomial Regression (degree={degree}): R^2 = {r2:.2f}, MSE = {mean_squared_error(Y, model.fittedvalues):.2f}')
            plt.legend()
            plt.grid()
            plt.show()
        
        # Step 5: 返回结果
        poly_result = {
            "Degree": degree,
            "F-statistic": f_statistic,
            "F-pvalue": f_pvalue,
            "R-squared": r2,
            "Adjusted R-squared": adj_r2,
            "AIC": aic,
            "BIC": bic
        }

        results = {
            'Degree': [degree],
            'R^2': [poly_result['R-squared']],
            'F-statistic': [poly_result['F-statistic']],
            'F-pvalue': [poly_result['F-pvalue']],
            'AIC': [poly_result['AIC']],
            'BIC': [poly_result['BIC']]
        }
        
        return pd.DataFrame(results)                    
