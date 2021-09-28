import os
import sys
import importlib
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import delay

#判断是否收盘处于涨跌停状态（即为不可交易状态）
def get_tradable_data(close):
    pre_close = delay(close, 1)
    up_limit = np.round(pre_close*1.1, decimals= 2)
    down_limit = np.round(pre_close*0.9, decimals= 2)
    return np.logical_and(close> down_limit,close< up_limit)

def get_index_ret(df_index_bars,dates):
    df_hs300 = df_index_bars[df_index_bars["order_book_id"] == "000300.XSHG"]
    df_hs300 = df_hs300.set_index("date").reindex(dates)
    df_hs300["ret"] = df_hs300["close"]/ df_hs300["close"].shift(1) -1 
    return df_hs300["ret"].values

def load_from_excel(path):
    df_price = pd.read_excel(r"{}\data.xlsx".format(path),sheet_name="price",header = 0, index_col= 0)
    df_mktv = pd.read_excel(r"{}\data.xlsx".format(path),sheet_name="mkt",header = 1, index_col= "Date")
    df_bv = pd.read_excel(r"{}\data.xlsx".format(path),sheet_name="bv",header = 1, index_col= "Date")

    df_bv = df_bv.reindex(df_mktv.index).fillna(method  ="bfill")

    data = {}
    data["close_adj"] = df_price.values
    data["ret"] = df_price.pct_change().values
    data["tradable"] = get_tradable_data(data["close_adj"])

    data["mktv"] = df_mktv.values
    data["bv"] = df_bv.values

    data["symbols"] = df_price.columns.values
    dates = df_price.index.values
    data["dates"] = np.array([str(d)[:10] for d in dates])
    data["index_ret"] = get_index_ret(pd.read_csv(r"{}\index_bars_1d.csv".format(path)),data["dates"])
    return data

#data loader
def load_from_npy():
    data = {}
    keys = os.listdir("data//.")
    print("[INFO]: loading data...")
    for key in keys:
        if key.endswith("npy"):
            data[key[:-4]] = np.load("data//{}".format(key))
    return data

#按天储存因子数据
def save_alpha(alpha_name,alpha_mat, symbols, dates):
    if not os.path.exists(f"output//{alpha_name}//alpha"):
        os.makedirs(f"output//{alpha_name}//alpha")
    for i,date in enumerate(dates):
        df_alpha = pd.DataFrame(alpha_mat[i,:],index = symbols, columns = ["factor"])
        df_alpha = df_alpha.dropna()
        df_alpha["date"] = date
        df_alpha = df_alpha.reset_index().rename(columns = {"index":"symbol"})
        path = f"output//{alpha_name}//alpha//{date}"
        df_alpha.to_csv(path)