import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt
import re
from datetime import datetime
from tqdm import tqdm

# Black-Scholes 公式的组成部分
def d1(S, strike_price, T, r, sigma):
    return (np.log(S / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, strike_price, T, r, sigma):
    return d1(S, strike_price, T, r, sigma) - sigma * np.sqrt(T)

# Delta
def delta(S, strike_price, T, r, sigma, option_type):
    if option_type == "认购":
        return norm.cdf(d1(S, strike_price, T, r, sigma))
    elif option_type == "认沽":
        return -norm.cdf(-d1(S, strike_price, T, r, sigma))

# Gamma
def gamma(S, strike_price, T, r, sigma):
    return norm.pdf(d1(S, strike_price, T, r, sigma)) / (S * sigma * np.sqrt(T))

# Vega
def vega(S, strike_price, T, r, sigma):
    return S * norm.pdf(d1(S, strike_price, T, r, sigma)) * np.sqrt(T)

# Theta
def theta(S, strike_price, T, r, sigma, option_type):
    if option_type == "认购":
        return - (S * norm.pdf(d1(S, strike_price, T, r, sigma)) * sigma) / (2 * np.sqrt(T)) - r * strike_price * np.exp(-r * T) * norm.cdf(d2(S, strike_price, T, r, sigma))
    elif option_type == "认沽":
        return - (S * norm.pdf(d1(S, strike_price, T, r, sigma)) * sigma) / (2 * np.sqrt(T)) + r * strike_price * np.exp(-r * T) * norm.cdf(-d2(S, strike_price, T, r, sigma))

def parse_asset_name(asset_name):
    match = re.search(r'(50ETF|300ETF|500ETF)(购|沽)(\d+)月(\d+)', asset_name)
    if match:
        option_type = '认购' if match.group(2) == '购' else '认沽'
        expiration_month = match.group(3)
        strike_price = match.group(4)
        return option_type, int(strike_price), expiration_month
    else:
        return None, None, None

def calculate_time_to_maturity(current_date, expiration_month):
    year = current_date.year
    maturity_date = datetime(year, int(expiration_month), 28)  # 假设每月到期日为月底
    days_to_maturity = (maturity_date - current_date).days
    return days_to_maturity / 365

# Black-Scholes公式
def black_scholes_call(S, strike_price, T, r, sigma):
    d1 = (np.log(S / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - strike_price * np.exp(-r * T) * norm.cdf(d2)

# 计算隐含波动率的函数
def implied_volatility(S, strike_price, T, r, market_price):
    def find_vol(sigma):
        return black_scholes_call(S, strike_price, T, r, sigma) - market_price
    try:
        return newton(find_vol, 0.2, tol=1e-10, maxiter=50)  # 调整初始估计、容忍度和最大迭代次数
    except RuntimeError:
        # 如果求解失败，可以返回一个默认值或进行其他处理
        return np.nan  # 例如，返回NaN



# 初始化变量
contracts = []
total_delta = 0
total_gamma = 0
total_vega = 0
total_theta = 0

# 标的资产价格范围
S_range = range(40, 70)

# 初始化整体收益列表
overall_profits = [0] * len(S_range)  # 初始化为0的列表

# 无风险利率
r = 0.015  # 1.5%
current_date = datetime.now()  # 当前日期

# 读取Excel文件
excel_data = pd.read_excel(r'E:\code\analysis report\期权持仓.xlsx')  # 替换为您的 Excel 文件路径

# 遍历Excel文件中的每一行
# 使用tqdm包装循环以显示进度条
for index, row in tqdm(excel_data.iterrows(), total=excel_data.shape[0]):
    # 解析ETF类型、行权价格、到期日和期权类型
    option_type, strike_price, expiration_month = parse_asset_name(row['资产名称'])
    direction = row['方向']  # 假设您的Excel表中有一个列名为'方向'，包含'买方'或'卖方'的值
    
    # 确保数据已正确解析
    if strike_price is not None:
        quantity = row['数量']
        market_price = row['市价']
        position_multiplier = 1 if direction == '买方' else -1  # 买方为正，卖方为负
        
        # 设置每个合约的特定参数
        S = market_price
        T = calculate_time_to_maturity(current_date, expiration_month)  # 假设每月到期日为月底        
        
        # 计算隐含波动率
        sigma = implied_volatility(S, strike_price, T, r, market_price)

        
        # 计算希腊字母值
        delta_val = delta(S, strike_price, T, r, sigma, option_type) * position_multiplier
        gamma_val = gamma(S, strike_price, T, r, sigma) * position_multiplier
        vega_val = vega(S, strike_price, T, r, sigma) * position_multiplier
        theta_val = theta(S, strike_price, T, r, sigma, option_type) * position_multiplier

        # 累加总希腊字母值
        total_delta += quantity * delta_val
        total_gamma += quantity * gamma_val
        total_vega += quantity * vega_val
        total_theta += quantity * theta_val

        # 计算整体收益
        for i, price in enumerate(S_range):
            profit = max(price - strike_price, 0) if option_type == "认购" else max(strike_price - price, 0)
            overall_profits[i] += quantity * profit * position_multiplier

        # 记录持仓
        contracts.append({
            "类型": option_type,
            "方向": direction,  # 添加方向信息
            "数量": quantity,
            "行权价格": strike_price,
            "市价": market_price,
            "Delta": delta_val,
            "Gamma": gamma_val,
            "Vega": vega_val,
            "Theta": theta_val
        })
        
# 计算希腊字母
delta_val = delta(S, strike_price, T, r, sigma, option_type)
gamma_val = gamma(S, strike_price, T, r, sigma)
vega_val = vega(S, strike_price, T, r, sigma)
theta_val = theta(S, strike_price, T, r, sigma, option_type)

# 绘制整体收益曲线
plt.plot(S_range, overall_profits)
plt.xlabel("标的资产价格")
plt.ylabel("整体收益")
plt.title("整体持仓收益曲线")
plt.show()

# 计算整体敞口水平
total_exposure = sum([contract['行权价格'] * contract['数量'] for contract in contracts])

# 打印持仓信息和总的希腊字母
print("持仓信息：")
for contract in contracts:
    print(contract)

print("总Delta:", total_delta)
print("总Gamma:", total_gamma)
print("总Vega:", total_vega)
print("总Theta:", total_theta)
print("整体敞口水平：", total_exposure)

# def monte_carlo_simulation(S0, mu, sigma, T, dt, N):
#     """
#     蒙特卡洛模拟计算波动率。

#     :param S0: 初始资产价格
#     :param mu: 预期收益率
#     :param sigma: 波动率
#     :param T: 总时间
#     :param dt: 时间步长
#     :param N: 模拟次数
#     :return: 模拟得到的波动率
#     """
#     M = int(T/dt)  # 总步数
#     price_paths = np.zeros((M + 1, N))
#     price_paths[0] = S0
    
#     # 生成随机价格路径
#     for t in range(1, M + 1):
#         z = np.random.standard_normal(N)  # 生成正态分布的随机数
#         price_paths[t] = price_paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    
#     # 计算波动率
#     log_returns = np.log(price_paths[1:] / price_paths[:-1])
#     simulated_volatility = np.std(log_returns) * np.sqrt(1/dt)
    
#     return simulated_volatility
