import os
import importlib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from functions import normalize,corr,delta, delay


#iterate through date
def daily_calcu_longshort(data,alpha_name, start_di = 0, end_di = 2000 ):

    alpha = importlib.import_module("research.{}".format(alpha_name))

    daily_ret_arr = np.full(fill_value = np.nan, shape= end_di - start_di )
    daily_turnover_arr = np.full(fill_value = np.nan, shape= end_di - start_di)
    alpha_mat = np.full(fill_value = np.nan, shape= (end_di - start_di,data["ret"].shape[1]))

    IC_arr = np.full(fill_value = np.nan, shape= (end_di - start_di,3) )

    ret = delta(data["close_adj"],1)/ delay(data["close_adj"], 1)
    for di in range(start_di+60 , end_di-1):
        print("[INFO]: backtest on {}...".format(data["dates"][di]))
        alpha_vec = alpha.generate( data, di)
        ret_vec = ret[di + 1] #第二天收盘交易
        
        alpha_vec = normalize( alpha_vec )
        ret_td = np.nansum(alpha_vec*ret_vec)
        
        IC_arr[di,0] = corr(alpha_vec, ret_vec)
        IC_arr[di,1] = corr(alpha_vec[alpha_vec>0], ret_vec[alpha_vec>0])
        IC_arr[di,2] = corr(alpha_vec[alpha_vec<0], ret_vec[alpha_vec<0])

        alpha_mat[di] = alpha_vec
        daily_ret_arr[di] = ret_td
        daily_turnover_arr[di] = np.nansum(np.abs(alpha_mat[di]-alpha_mat[di-1]))
     
    return daily_ret_arr, daily_turnover_arr, alpha_mat

def daily_calcu_IC_hedge(data,alpha_name, start_di = 0, end_di = 2000 ):

    alpha = importlib.import_module("research.{}".format(alpha_name))

    daily_ret_arr = np.full(fill_value = np.nan, shape= end_di - start_di )
    daily_turnover_arr = np.full(fill_value = np.nan, shape= end_di - start_di)
    alpha_mat = np.full(fill_value = np.nan, shape= (end_di - start_di,data["ret"].shape[1]))

    ret =  delta(data["close_adj"],1)/ delay(data["close_adj"], 1)
    for di in range(start_di+60 , end_di-1):
        print("[INFO]: backtest on {}...".format(data["dates"][di]))
        alpha_vec = alpha.generate( data, di)
        ret_vec = ret[di + 1]#第二天收盘交易
        
        alpha_vec = normalize( alpha_vec )
        alpha_vec = 2* np.where(alpha_vec<0, 0, alpha_vec)

        index_ret = data["index_ret"][di+1]
        ret_td = np.nansum(alpha_vec*ret_vec) - index_ret

        alpha_mat[di] = alpha_vec
        daily_ret_arr[di] = ret_td
        daily_turnover_arr[di] = np.nansum(np.abs(alpha_mat[di]-alpha_mat[di-1]))
     
    return daily_ret_arr, daily_turnover_arr, alpha_mat

#calculate and plot the result
def summary_calcu(daily_ret_arr, daily_turnover_arr,data, start_di = 0, end_di = 2000):
    print("[INFO]: summarizing...")
    overall_sharpe = np.nanmean(daily_ret_arr) / np.nanstd(daily_ret_arr) * np.sqrt(252)
    overall_return = np.nansum(daily_ret_arr) / (end_di - start_di) * 252
    overall_turnover = np.nanmean(daily_turnover_arr) 

    # performance in each year
    performance_each_year= np.full(fill_value = np.nan, shape= (9, 3))
    for year in range(2013,2021):
        di_range = np.logical_and(data["dates"].astype(str)> "{}0101".format(year), data["dates"].astype(str)< "{}0101".format(year+1))
        ret_arr_ty = daily_ret_arr[di_range]
        turnover_arr_ty = daily_turnover_arr[di_range]

        sharpe_ty = np.nanmean(ret_arr_ty) / np.nanstd(ret_arr_ty) * np.sqrt(252)
        return_ty = np.nansum(ret_arr_ty)
        turnover_ty = np.nanmean(turnover_arr_ty)

        performance_each_year[year-2013] = np.array([turnover_ty,return_ty,sharpe_ty])
    
    performance_each_year[-1] = np.array([overall_turnover ,overall_return, overall_sharpe])
    
    performance_each_year = pd.DataFrame(performance_each_year, index = list(range(2013,2021)) + ["overall"], columns = ["turnover", "return", "sharpe"])
    return performance_each_year
    


