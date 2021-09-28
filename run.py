import os
import sys
import importlib
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import delay
from utils import get_tradable_data,get_index_ret,save_alpha,load_from_excel

def dummy_performance_longshort(data,alpha_name,start_di,end_di):
    """
    以因子值为权重构建多空组合，因子值排名前50%为多头，后50%为空头，测试组合表现
    输出图表说明：
    1.longshort_cumret_plot:因子多空组合累计收益图
    2.longshort_performance: 因子多空组合每年表现以及所以年总表现
    3.longshort_ret: 因子多空组合每日收益率
    """
    symbols, dates= data["symbols"],data["dates"]
    dummy_performance = importlib.import_module("dummy_performance")
    daily_ret_arr, daily_turnover_arr, alpha_mat = dummy_performance.daily_calcu_longshort(data,alpha_name, start_di = start_di, end_di = end_di)
    performance = dummy_performance.summary_calcu(daily_ret_arr, daily_turnover_arr, data, start_di = start_di, end_di = end_di)
    save_alpha(alpha_name,alpha_mat,symbols, dates[start_di:end_di])

    daily_ret = pd.DataFrame(daily_ret_arr, index = data["dates"][start_di:end_di], columns =["return"])

    if not os.path.exists("output//{}//dummy_performance".format(alpha_name)):
        os.makedirs("output//{}//dummy_performance".format(alpha_name))

    plt.grid()
    plt.plot( pd.to_datetime(dates.astype(str)[start_di:end_di]), np.nancumsum(daily_ret_arr) )

    plt.savefig("output//{}//dummy_performance//{}_longshort_cumret_plot.jpg".format(alpha_name,alpha_name),dpi = 200)
    plt.cla()
    daily_ret.to_csv("output//{}//dummy_performance//{}_longshort_ret.csv".format(alpha_name,alpha_name),index_label= "date")
    performance.to_csv("output//{}//dummy_performance//{}_longshort_performance.csv".format(alpha_name,alpha_name),index_label= "year")

def dummy_performance_hedge(data,alpha_name,start_di,end_di):
    """
    以因子值为权重构建对冲组合，因子值排名前50%为多头，后50%改为指数对冲，测试组合表现
    输出图表说明：
    1.hedge_cumret_plot:因子对冲组合累计收益图
    2.hedge_performance: 因子对冲组合每年表现以及所以年总表现
    3.hedge_ret: 因子对冲组合每日收益率
    """
    symbols, dates= data["symbols"],data["dates"]
    dummy_performance = importlib.import_module("dummy_performance")
    daily_ret_arr, daily_turnover_arr, alpha_mat = dummy_performance.daily_calcu_IC_hedge(data,alpha_name, start_di = start_di, end_di = end_di)
    performance = dummy_performance.summary_calcu(daily_ret_arr, daily_turnover_arr, data, start_di = start_di, end_di = end_di)
    save_alpha(alpha_name,alpha_mat,symbols, dates[start_di:end_di])

    daily_ret = pd.DataFrame(daily_ret_arr, index = data["dates"][start_di:end_di], columns =["return"])

    plt.grid()
    plt.plot( pd.to_datetime(dates.astype(str)[start_di:end_di]), np.nancumsum(daily_ret_arr) )

    if not os.path.exists("output//{}//dummy_performance".format(alpha_name)):
        os.makedirs("output//{}//dummy_performance".format(alpha_name))
    plt.savefig("output//{}//dummy_performance//{}_hedge_cumret_plot.jpg".format(alpha_name,alpha_name),dpi = 200)
    plt.cla()
    daily_ret.to_csv("output//{}//dummy_performance//{}_hedge_ret.csv".format(alpha_name,alpha_name),index_label= "date")
    performance.to_csv("output//{}//dummy_performance//{}_hedge_performance.csv".format(alpha_name,alpha_name),index_label= "year")

def layer_performance(data,alpha_name,start_di,end_di):
    """
    按照因子值进行分层测试，观察每层表现（经过指数对冲后）
    输出图表说明：
    1.layer_bars:每层收益率以及夏普比条形图
    2.layer_performance: 每层累计收益图
    3.perf_each_layer: 每层收益率、夏普比、换手率统计表
    4.top_layer_perf: 因子值最大层分年的表现统计（收益夏普换手）

    """
    symbols, dates= data["symbols"],data["dates"]
    layer_performance = importlib.import_module("layer_performance")
    daily_ret_arr, daily_turnover_arr,alpha_mat = layer_performance.daily_calcu(data,alpha_name,  start_di = start_di, end_di = end_di)
    performance_eachlayer,performance_top_each_year = layer_performance.summary_calcu(daily_ret_arr,daily_turnover_arr, data, start_di = start_di, end_di = end_di)
    save_alpha(alpha_name,alpha_mat,symbols, dates[start_di:end_di])

    daily_cumret_arr = np.nancumsum(daily_ret_arr, axis = 0)
    df_daily_cumret = pd.DataFrame(daily_cumret_arr,
        index = pd.to_datetime(dates.astype(str)[start_di:end_di]),
        columns = ["group"+str(i) for i in range(1, 11)] )
    
    if not os.path.exists("output//{}//layer_performance".format(alpha_name)):
        os.makedirs("output//{}//layer_performance".format(alpha_name))
    df_daily_cumret.plot();plt.legend();plt.grid()
    plt.savefig("output//{}//layer_performance//{}_layer_performance.jpg".format(alpha_name,alpha_name),dpi = 200)
    plt.cla()
    plt.subplot(2,1,1)
    performance_eachlayer["sharpe"].plot(kind = "bar")
    plt.subplot(2,1,2)
    performance_eachlayer["return"].plot(kind = "bar")
    plt.savefig("output//{}//layer_performance//{}_layer_bars.jpg".format(alpha_name,alpha_name),dpi = 200)


    performance_top_each_year.to_csv("output//{}//layer_performance//{}_top_layer_perf.csv".format(alpha_name,alpha_name), index_label = "year")
    performance_eachlayer.to_csv("output//{}//layer_performance//{}_perf_each_layer.csv".format(alpha_name,alpha_name),index_label= "layer")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-a',dest='alpha')
    parser.add_option('-s',dest='start_date', default = "2019-07-01")
    parser.add_option('-e',dest='end_date', default = "2021-07-02")
    parser.add_option('--type',dest='bt_type', default = "layer_performance")
    (options, args) = parser.parse_args()

    data = load_from_excel(r"data")
    
    symbols = data["symbols"]
    dates = data["dates"]
    start_di = np.searchsorted(dates,options.start_date ) 
    end_di = np.searchsorted(dates,options.end_date ) 

    if options.bt_type == "dummy_performance":
        dummy_performance_longshort(data,options.alpha,start_di,end_di)

        dummy_performance_hedge(data,options.alpha,start_di,end_di)

    elif options.bt_type == "layer_performance":
        layer_performance(data,options.alpha,start_di,end_di)

