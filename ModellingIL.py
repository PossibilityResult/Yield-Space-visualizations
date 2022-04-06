from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

api_key = 'aE360XwCWkNSXn8pONYTQzuKo5oTGHF4eo5jeQ9FGAwQmlzOO0bdXi394xARz9Z1'
api_secret = '8Bh4vjBXK9zFkBgWgD8xv3i55Z8CS7G0wsDmNlk3yzp8JuJMzo8FlWNVRQl4LDtb'
client = Client(api_key, api_secret)
dt = 1 / (365*24)


# https://quant.stackexchange.com/questions/55220/simulate-stock-prices-with-geometric-brownian-motion-motion-with-mu-and-signa-ba
# Get historical log normal average returns
def get_mu_bar(prices: pd.Series) -> float:
    log_returns = np.log(prices) - np.log(prices.shift(1))
    mu_bar = np.sum(log_returns) / log_returns.size
    return mu_bar


# Get historical volatility
def get_sigma_bar(mu_bar: float, prices: pd.Series) -> float:
    log_returns = np.log(prices) - np.log(prices.shift(1))
    sigma_sq_bar = np.sum((log_returns - mu_bar) ** 2) / (log_returns.size - 1)
    return np.sqrt(sigma_sq_bar)


# Get volatility on a daily time frame
def get_sigma(sigma_bar: float) -> float:
    return sigma_bar / np.sqrt(dt)


# Get the drift of the asset
def get_mu(mu_bar: float, sigma: float) -> float:
    return mu_bar / dt + .5 * sigma ** 2


# Get drift of the assets by passing in the data
def get_mu_from_prices(closes_in: pd.Series) -> float:
    mu_bar = get_mu_bar(closes_in)
    sigma_bar = get_sigma_bar(mu_bar, closes_in)
    sigma = get_sigma(sigma_bar)
    mu_out = get_mu(mu_bar, sigma)
    return mu_out


# Get asset prices
def get_prices(asset_list: list, date: str) -> pd.DataFrame:
    df = pd.DataFrame()

    # Get all the closing prices for assets in USDT
    for symbol in asset_list:
        klines = client.get_historical_klines(f'{symbol}USDT', Client.KLINE_INTERVAL_1HOUR, date)
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        # data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        df[symbol] = data['close'].astype(float)

    return df

def get_z_prices(df: pd.DataFrame, numeraire: str) -> pd.DataFrame:
    # Normalize asset prices in terms of the numeraire
    df_z = pd.DataFrame()

    for symbol in df:
        df_z[symbol] = df[symbol] / df[numeraire]

    return df_z


def projected_z_prices(z_prices_df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    symbol_mu_tuples = []
    for column in z_prices_df:
        series = z_prices_df[column]
        symbol_mu_tuples.append((column, get_mu_from_prices(series)))

    df = pd.DataFrame()
    for symbol, mu in symbol_mu_tuples:
        x0 = get_current_price(symbol) / get_current_price('ETH')
        proj_price = []
        for i in range(iterations):
            proj_price.append(x0*np.exp(mu*i*dt))
        df[symbol] = proj_price
    return df


def get_current_price(symbol: str) -> float:
    current_price = float(client.get_avg_price(symbol=f'{symbol}USDT')['price'])
    return current_price


def projected_numeraire_price(prices: pd.Series, iterations: int) -> pd.Series:
    mu = get_mu_from_prices(prices)
    x0 = get_current_price(prices.name)

    prices_list = []
    for i in range(iterations):
        prices_list.append(x0 * np.exp(mu * i * dt))

    return pd.Series(prices_list)


def convert_IL_to_dollars(IL_z: pd.DataFrame, numeraire_prices: pd.Series, initial_pool_value) -> pd.DataFrame:
    return IL_z * numeraire_prices


def get_initial_prices(asset_list: list, date: str) -> list:
    initial_prices = []
    for symbol in asset_list:
        klines = client.get_historical_klines(f'{symbol}USDT', Client.KLINE_INTERVAL_1DAY, date)
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        # data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        prices = data['close'].astype(float)
        initial_prices.append(prices.iloc[0])
    return initial_prices


def get_IL(df: pd.DataFrame, asset_list: list, initial_pool_value=10000, weights=[]):
    n = len(asset_list)
    if not weights:
        weights = [1/n for _ in range(n)]

    initial_prices = get_initial_prices(asset_list)
    initial_reserves = np.array([initial_pool_value / n / price for price in initial_prices])

    # HODL Reserve Value: Sum of r_i * p_i'
    hodl_values = df.dot(initial_reserves)

    # LP Reserve Value: Product of (r_i * p_i')^n / n
    lp_values = n * (df.prod(axis=1) * initial_reserves.prod(axis=0)) ** (1/n)

    impermanent_loss = hodl_values - lp_values

    return impermanent_loss

def calculate_historical_IL(df: pd.DataFrame, asset_list: list, initial_pool_value=10000):
    n = len(asset_list)

    initial_prices = list(df.iloc[0])
    print(initial_prices)
    initial_reserves = np.array([initial_pool_value / n / price for price in initial_prices])

    # HODL Reserve Value: Sum of r_i * p_i'
    hodl_values = df.dot(initial_reserves)

    # LP Reserve Value: Product of (r_i * p_i')^n / n
    lp_values = n * (df.prod(axis=1) * initial_reserves.prod(axis=0)) ** (1/n)

    impermanent_loss = hodl_values - lp_values

    return impermanent_loss

def plot_impermanent_loss(impermanent_loss: pd.Series):
    plt.plot(impermanent_loss)
    plt.show()

def get_impermanent_loss(asset_list: list, iterations=365*24, initial_pool_value=1):
    df_prices = get_prices(asset_list)
    iterations = 365 * 24

    z_prices_project = projected_z_prices(df_prices, iterations)

    IL = get_IL(z_prices_project, asset_list)

    return(IL)

def get_historical_il(asset_list: list, date: str, iterations=365*24, initial_pool_value=1):
    df_prices = get_prices(asset_list, date)
    iterations = 365 * 24

    IL = calculate_historical_IL(df_prices, asset_list)

    return IL

def main():
    asset_list = ['ETH', 'BTC']
    IL = get_historical_il(asset_list, '1 Mar 2021')
    plt.plot(IL)
    plt.show()

if __name__ == '__main__':
    main()
