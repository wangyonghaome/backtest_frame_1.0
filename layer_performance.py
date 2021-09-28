import os
import importlib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from functions import normalize, score,delta, delay


#iterate through date
def daily_calcu(data, alpha_name, layer_num = 10, start_di = 0, end_di = 2000 ):

    alpha = importlib.import_module("research.{}".format(alpha_name))

    daily_ret_arr = np.full(fill_value = np.nan, shape= (end_di - start_di, layer_num) )
    daily_turnover_arr = np.full(fill_value = np.nan, shape= (end_di - start_di, layer_num) )
    alpha_mat = np.full(fill_value = np.nan, shape= (end_di - start_di,data["ret"].shape[1]))

    ret =  delta(data["close_adj"],1)/ delay(data["close_adj"], 1)
    for di in range(start_di+60 , end_di-1):
        print("[INFO]: backtest on {}...".format(data["dates"][di]))
        alpha_vec = alpha.generate( data, di)
        ret_vec = ret[di + 1]#第二天收盘交易
        
        alpha_mat[di] = alpha_vec
        stknum_each = int(np.nansum(~np.isnan(data["ret"][di])) / layer_num)
        alpha_score = score(alpha_vec, layer= layer_num)
        alpha_score_pre = score(alpha_mat[di-1], layer= layer_num)

        index_ret = data["index_ret"][di + 1]
        for layer in range(layer_num):
            daily_turnover_arr[di,layer] = np.nansum(np.abs((alpha_score==layer).astype(int) -
                                         (alpha_score_pre==layer).astype(int))/stknum_each)
            daily_ret_arr[di,layer] = np.nanmean(ret_vec[alpha_score==layer] - index_ret)
     
    return daily_ret_arr,daily_turnover_arr, alpha_mat

#calculate and plot the result
def summary_calcu(daily_ret_arr,daily_turnover_arr,data, layer_num =10, start_di = 0, end_di = 2000):
    print("[INFO]: summarying...")

    # performance in each year
    performance_top_each_year= np.full(fill_value = np.nan, shape= (9, 3))
    for year in range(2013,2021):
        #TO DO 
        di_range = np.logical_and(data["dates"]> "{}-01-01".format(year), data["dates"]< "{}-01-01".format(year+1))
        if np.nansum(di_range)== 0:
            continue
        ret_arr_ty = daily_ret_arr[di_range,-1]
        turnover_arr_ty = daily_turnover_arr[di_range,-1]
        performance_top_each_year[year-2013,0] = np.nanmean(ret_arr_ty)*252
        performance_top_each_year[year-2013,1] = np.nanmean(ret_arr_ty) / np.nanstd(ret_arr_ty) * np.sqrt(252)
        performance_top_each_year[year-2013,2] = np.nanmean(turnover_arr_ty)

    performance_each_layer =  np.full(fill_value = np.nan, shape= (layer_num, 3))
    for layer in range(layer_num):
        performance_each_layer[layer,0] = np.nanmean(daily_ret_arr[:,layer])*252
        performance_each_layer[layer,1] = np.nanmean(daily_ret_arr[:,layer])/np.nanstd(daily_ret_arr[:,layer]) * np.sqrt(252)
        performance_each_layer[layer,2] = np.nanmean(daily_turnover_arr[:,layer])
    
    performance_top_each_year[-1,0] = np.nanmean(daily_ret_arr[:,-1])*252
    performance_top_each_year[-1,1] = np.nanmean(daily_ret_arr[:,-1])/np.nanstd(daily_ret_arr[:,-1]) * np.sqrt(252)
    performance_top_each_year[-1,2] = np.nanmean(daily_turnover_arr[:,-1])

    performance_top_each_year = pd.DataFrame(performance_top_each_year, index = list(range(2013,2021)) + ["overall"], columns = [ "return", "sharpe","turnover"])
    performance_each_layer = pd.DataFrame(performance_each_layer, index= np.arange(layer_num), columns = [ "return", "sharpe","turnover"])

    return performance_each_layer,performance_top_each_year
    


