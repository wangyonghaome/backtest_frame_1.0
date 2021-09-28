import numpy as np
import math

def rank(arr,axis =0 ):
    if np.ndim(arr) == 1:
        return arr.argsort().argsort()
    return arr.argsort(axis =axis).argsort(axis =axis)

def corr(arr_1,arr_2):
    arr_1 = arr_1.copy().astype(float)
    arr_2 = arr_2.copy().astype(float)
    arr_1[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan
    arr_2[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan

    return  (np.nansum((arr_1- np.nanmean(arr_1,axis=0 ))*(arr_2 - np.nanmean(arr_2,axis =0)),axis = 0) /
              np.sqrt(np.nansum((arr_2- np.nanmean(arr_2,axis =0))**2,axis =0)*np.nansum((arr_1- np.nanmean(arr_1,axis =0))**2,axis =0)))


def rank_corr(arr_1,arr_2):
    return corr(rank(arr_1),rank(arr_2))

def ols(arr_1,arr_2):
    arr_1 = arr_1.copy()
    arr_2 = arr_2.copy()
    arr_1[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan
    arr_2[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan

    beta = np.nansum((arr_1- np.nanmean(arr_1,axis=0 ))*(arr_2 - np.nanmean(arr_2,axis =0)),axis = 0) / np.nansum((arr_2- np.nanmean(arr_2,axis =0))**2,axis =0)
    alpha = np.nanmean(arr_1,axis=0) - beta*np.nanmean(arr_2,axis=0)
    resid = arr_1 - alpha - beta*arr_2
    return {"alpha":alpha ,"beta": beta,"resid": resid}

def wls(arr_1,arr_2,half_period,lamb = 2):
    arr_1 = arr_1.copy()
    arr_2 = arr_2.copy()
    arr_1[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan
    arr_2[np.isnan(arr_1) | np.isnan(arr_2)] = np.nan

    weight = np.power(math.pow(lamb,1.0/half_period),np.arange(arr_1.shape[0]))
    weight = weight/np.sum(weight)
    arr_1 = arr_1*repeat(weight,arr_1.shape[1])
    arr_2 = arr_2*repeat(weight,arr_2.shape[1])

    beta = np.nansum((arr_1- np.nanmean(arr_1,axis=0 ))*(arr_2 - np.nanmean(arr_2,axis =0)),axis = 0) / np.nansum((arr_2- np.nanmean(arr_2,axis =0))**2,axis =0)
    alpha = np.nanmean(arr_1,axis=0) - beta*np.nanmean(arr_2,axis=0)
    resid = ((arr_1 - alpha - beta*arr_2).T/weight).T
    return {"alpha":alpha ,"beta": beta,"resid": resid}


def down_beta(arr_1, arr_2, method= "mean"):
    down_arr_1 = arr_1.copy()
    down_arr_2 = arr_2.copy()
    if method == "median" :
        down_arr_1[down_arr_2> np.nanmedian(down_arr_2,axis = 0)] = np.nan
        down_arr_2[down_arr_2> np.nanmedian(down_arr_2,axis = 0)] = np.nan
    if method == "mean" :
        down_arr_1[down_arr_2> np.nanmean(down_arr_2,axis = 0)] = np.nan
        down_arr_2[down_arr_2> np.nanmean(down_arr_2,axis = 0)] = np.nan
    return ols(down_arr_1,down_arr_2)["beta"]

def up_beta(arr_1,arr_2, method = "mean"):
    up_arr_1 = arr_1.copy()
    up_arr_2 = arr_2.copy()
    if method == "median" :
        up_arr_1[up_arr_2< np.nanmedian(up_arr_2,axis = 0)] = np.nan
        up_arr_2[up_arr_2< np.nanmedian(up_arr_2,axis = 0)] = np.nan
    if method == "mean" :
        up_arr_1[up_arr_2< np.nanmean(up_arr_2,axis = 0)] = np.nan
        up_arr_2[up_arr_2< np.nanmean(up_arr_2,axis = 0)] = np.nan
    return ols(up_arr_1, up_arr_2)["beta"]

def upvolat(arr,axis =0):
    mu = np.nanmean(arr,axis =0)
    arr_up = arr.copy()
    arr_up[arr_up<mu] = np.nan
    return np.nanmean((arr_up-mu)**2,axis =0)

def dnvolat(arr, axis =0):
    mu = np.nanmean(arr,axis =0)
    arr_dn = arr.copy()
    arr_dn[arr_dn>mu] = np.nan
    return np.nanmean((arr_dn-mu)**2,axis =0)

def delay(arr,d):
    return np.concatenate([np.full((d,arr.shape[1]),np.nan),arr[:-d,:]])

def delta(arr,d):
    return np.concatenate([np.full((d,arr.shape[1]),np.nan),arr[d:,:]-arr[:-d:,:]]) 

def repeat(arr,n):
    return arr.repeat(n).reshape((len(arr),n))


def trend(arr,p1,p2):
    return np.nanmean(arr[-p1:,:],axis=0)- np.nanmean(arr[-p2:,:],axis=0)

def stddev_trend(arr,p1,p2):
    return np.nanstd(arr[-p1:,:],axis=0)- np.nanstd(arr[-p2:,:],axis=0)

def ewa(arr,half_period,keepdims=False,lamb=2):
    weight = np.power(math.pow(lamb,1.0/half_period),np.arange(arr.shape[0]))
    weight = weight/np.sum(weight)
    return np.nansum(arr*repeat(weight,arr.shape[1]), axis = 0,keepdims=keepdims)

def ewstd(arr,half_period,keepdims=False,lamb=2):
    weight = np.power(math.pow(lamb,1.0/half_period),np.arange(arr.shape[0]))
    weight = weight/np.sum(weight)
    return np.sqrt(np.nansum(((arr - ewa(arr,half_period,lamb=lamb))**2)*repeat(weight,arr.shape[1]), axis = 0,keepdims=keepdims))

def rolling_mean(arr,window):
    arr_ls = [np.nanmean(arr[i:i+window,:], axis=0, keepdims = True) for i in range(arr.shape[0]-window+1) ]
    return np.concatenate(arr_ls, axis=0)

def rolling_stddev(arr,window):
    arr_ls = [np.nanstd(arr[i:i+window,:], axis=0, keepdims = True) for i in range(arr.shape[0]-window+1) ]
    return np.concatenate(arr_ls, axis=0)

def rolling_ewa(arr,window,half_period):
    arr_ls = [ewa(arr[i:i+window,:],half_period=half_period, keepdims = True, lamb=2) for i in range(arr.shape[0]-window+1) ]
    return np.concatenate(arr_ls, axis=0)

def rolling_ewstd(arr,window,half_period):
    arr_ls = [ewstd(arr[i:i+window,:],half_period=half_period, keepdims = True, lamb=2) for i in range(arr.shape[0]-window+1) ]
    return np.concatenate(arr_ls, axis=0)

def ts_stddize(arr):
    return (arr[-1]-np.nanmean(arr,axis=0))/ np.nanstd(arr, axis=0)

def sigmoid(arr):
    return 1/(1+np.exp(-1*arr))

def score(arr, layer = 10):
    if np.ndim(arr) ==1 :
        score_arr = np.full(fill_value = np.nan, shape = len(arr))
        score_arr[~np.isnan(arr)] =  np.floor(rank(arr[~np.isnan(arr)],axis=1)*layer/len(arr))
        return score_arr
    stk_num = arr.shape[1]
    rank_arr = rank(arr,axis=1)
    score_arr = np.floor(rank_arr*layer/stk_num)
    score_arr[np.isnan(arr)] = np.nan
    return score_arr

def indus_neutralize(arr, indus_dummy):
    indus_range = np.unique(indus_dummy)
    for indus in indus_range:
        arr_ind = arr[indus_dummy == indus]
        if len(arr_ind)==0:
            continue
        arr[indus_dummy == indus] = (arr_ind- np.nanmean(arr_ind)) / np.nanstd(arr_ind)
    return arr

def cap_neutralize(arr , nego_cap, layer = 10):
    nego_cap_score = score(nego_cap, layer = layer)
    for i in range(layer):
        arr_cap = arr[nego_cap_score == i]
        if len(arr_cap)==0:
            continue
        arr[nego_cap_score == i] = (arr_cap- np.nanmean(arr_cap)) / np.nanstd(arr_cap)
    return arr

def normalize(arr):
    arr = (arr- np.nanmean(arr)) / np.nanstd(arr)
    arr[arr>0] =  0.5* (arr[arr>0])/np.nansum(arr[arr>0])
    arr[arr<0] = -0.5* (arr[arr<0])/np.nansum(arr[arr<0])
    return arr