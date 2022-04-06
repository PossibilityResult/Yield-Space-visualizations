from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

api_key = 'aE360XwCWkNSXn8pONYTQzuKo5oTGHF4eo5jeQ9FGAwQmlzOO0bdXi394xARz9Z1'
api_secret = '8Bh4vjBXK9zFkBgWgD8xv3i55Z8CS7G0wsDmNlk3yzp8JuJMzo8FlWNVRQl4LDtb'
client = Client(api_key, api_secret)

# Get asset prices
def get_prices(asset_list: list, date: str) -> pd.DataFrame:
    df = pd.DataFrame()

    # Get all the closing prices for assets in USDT
    for symbol in asset_list:
        klines = client.get_historical_klines(f'{symbol}USDT', Client.KLINE_INTERVAL_1DAY, date)
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        # data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        df[symbol] = data['close'].astype(float)
    df['date'] = pd.to_datetime(data['timestamp'], unit='ms')
    return df.set_index('date', inplace=False)

def get_current_price(asset_list: list) -> list:
    current_price = [float(client.get_avg_price(symbol=f'{symbol}USDT')['price']) for symbol in asset_list]
    return current_price

def calculate_IL(asset_list: list, date, initial_pool_value=1):
    initial_prices = get_prices(asset_list, date)
    n = len(asset_list)
    initial_reserves = []

    for _, row in initial_prices.iterrows():
        reserve_row = []
        for asset in asset_list:
            reserve_row.append(initial_pool_value / n / row[asset])
        initial_reserves.append(reserve_row)

    current_prices = get_current_price(asset_list)

    # HODL Reserve Value: Sum of r_i * p_i'
    hodl_values = np.dot(np.array(initial_reserves), current_prices)

    # LP Reserve Value: Product of (r_i * p_i')^n / n
    lp_values = n * (np.array(initial_reserves).prod(axis=1) * np.array(current_prices).prod(axis=0)) ** (1/n)

    impermanent_loss = hodl_values - lp_values

    name = '-'.join(asset_list)

    df = initial_prices.copy()[[]]
    # df['date'] = initial_prices['date']
    df[name] = impermanent_loss
    # df2 = df.set_index('date', inplace=False)
    return df, name

def il_date_invested():
    df, _ = calculate_IL(['ETH', 'BTC'], '1 Jan 2021')
    asset_lists = [['USDC', 'ETH'], ['ETH', 'SUSHI'], ['ETH', 'UNI'], ['USDC', 'ETH', 'BTC']]
    for asset_list in asset_lists:
        df2, name = calculate_IL(asset_list, '1 Jan 2021')
        df[name] = df2[name]
    df.plot()
    plt.title('Impermanent Loss vs Initial Liquidity Provided')
    plt.show()

def calculate_historical_il(asset_list: list):
    n = len(asset_list)
    initial_pool_value = 1
    df = get_prices(asset_list, '1 Jan 2021')
    initial_prices = list(df.iloc[0])
    initial_reserves = np.array([initial_pool_value / n / price for price in initial_prices])

    # HODL Reserve Value: Sum of r_i * p_i'
    hodl_values = df.dot(initial_reserves)

    # LP Reserve Value: Product of (r_i * p_i')^n / n
    lp_values = n * (df.prod(axis=1) * initial_reserves.prod(axis=0)) ** (1/n)

    impermanent_loss = hodl_values - lp_values

    name = '-'.join(asset_list)
    df2 = pd.DataFrame()
    df2[name] = impermanent_loss
    return df2

def historical_il():
    df = calculate_historical_il(['ETH', 'BTC'])
    asset_lists = [['USDC', 'ETH'], ['ETH', 'SUSHI'], ['ETH', 'UNI'], ['USDC', 'ETH', 'BTC']]
    for asset_list in asset_lists:
        df2 = calculate_historical_il(asset_list)
        for column in df2:
            df[column] = df2[column]

    df.plot()
    plt.title('Historical Impermanent Loss')
    plt.show()

def main():
    il_date_invested()

if __name__ == '__main__':
    main()
