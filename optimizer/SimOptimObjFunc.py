'''
Created on Dec 6, 2020

@author: smin
'''
'''
Created on Mar 6, 2019

@author: smin
'''
import math
import numpy as np
import pandas as pd

from IPython.display import display
pd.options.display.max_colwidth=500

def _validity(value):
    if value is None or np.isnan(value) or np.isinf(value):
        return False
    else:
        return True


def CalcSharpe(eventSum_df):
    if eventSum_df.shape[0] == 0:
        return '',np.nan,True

    stgy_name = eventSum_df.iloc[0]['ticker_id']
    if eventSum_df.shape[0] < 1:
        return stgy_name, np.nan, True
    
    stgy_name = eventSum_df.iloc[0]['ticker_id'] 
    mean = eventSum_df['ave_pnl'].mean()
    stdvalue = np.nanstd(eventSum_df['ave_pnl'].values)
    score = mean/stdvalue if _validity(stdvalue) else np.nan
    score = score if _validity(score) else float('nan')
    return stgy_name, score, True

def CalcMean(eventSum_df):
    if eventSum_df.shape[0] == 0:
        return '', np.nan

    stgy_name = eventSum_df.iloc[0]['ticker_id']
    score = np.nanmean(eventSum_df['ave_pnl'].values)
    score = score if _validity(score) else float('nan')
    return stgy_name, score, True

def CalcMWR(eventSum_df):
    if eventSum_df.shape[0] == 0:
        return np.nan

    stgy_name = eventSum_df.iloc[0]['ticker_id']
    wr = (1+ np.sum(eventSum_df['ave_pnl'].values > 0) / float(len(eventSum_df['ave_pnl'].values))) * 100
    
    score = wr
    score = score if _validity(score) else np.nan
    return stgy_name, score, True

def Calc_SM1(eventSum_df):
    stgy_name = eventSum_df.iloc[0]['ticker_id']
    pnl = eventSum_df['ave_pnl'].mean()
    wr = eventSum_df['win_rate'].mean()
    aqty = eventSum_df['tot_qty'].mean()
    
    if pnl > 0.5:
        score = wr * 10 * pnl
    else:
        score = pnl
    
    score = score if _validity(score) else np.nan
    return stgy_name, score, True


def normalize(profit,lbound, ubound):
    rng = ubound-lbound
    profit = profit
    norm = rng / (1 + math.exp(-1/rng * profit)) + lbound
    return norm

def Calc_SM0(eventSum_df):
    if eventSum_df.shape[0] == 0:
        return np.nan

    stgy_name = eventSum_df.iloc[0]['ticker_id']
    sharpe = eventSum_df['sharpe'].iloc[0]
    wr = eventSum_df['win_rate'].iloc[0]
    aqty = eventSum_df['tot_qty'].iloc[0]
    xqty = 10 / (1 + math.exp(-0.002 * aqty))    
    
    score = wr * 10 + sharpe + xqty
    if sharpe < 0:
        score = sharpe
    elif sharpe > 5.0:
        score = wr * 10 + xqty
    score = score if _validity(score) else np.nan
    return stgy_name, score, True


obj_functions = {
     'max_sharpe': CalcSharpe
    ,'max_pnl'   : CalcMean
    ,'maxWR'     : CalcMWR
    ,'SM1'       : Calc_SM1
    ,'SM0'       : Calc_SM0
}


