import numpy as np 

from functions import *

# 利用di-prev_n到di-1天的数据计算di天的因子值，di天收盘进行交易
def generate(data, di):
    prev_n = 20

    ix = (data['tradable'][di , :] == 1)# 确保第二天交易的时候不处于涨跌停状态
    close = data["close_adj"][di-prev_n:di,ix]
    
    alpha_vec = np.full(fill_value=np.nan, shape= data["tradable"].shape[1])
    alpha_vec[ix] = -(close[-1] / np.nanmean(close, axis = 0) - 1)*100
    return alpha_vec
