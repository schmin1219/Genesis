'''
Created on Dec 27, 2020

@author: smin
'''
import os
import json
import time
import pandas as pd
from datetime import datetime

class CommonDir(object):    
    repo_dir = 'c:\\Users\schmi\workspace\StockTradingBot'
    result_dir = 'c:\\Users\schmi\IBStockEngine'
    common_dir = os.path.join(repo_dir, 'common')
    train_output_dir = os.path.join(result_dir,'result')
    sim_output_dir = os.path.join(result_dir, 'sim')
    sampled_dir = os.path.join(result_dir, 'data')
    sampled_us_dir = os.path.join(result_dir, 'us_data')
    portfolio_dir = os.path.join(result_dir, 'portfolio')
    trading_dir = os.path.join(result_dir, 'trading')
    
    def __init__(self):
        pass
    
def jsonToname(json_):
    str_ = json.dumps(json_)
    str_ = str_.replace('"', '').replace('{','').replace('}','').replace(':','')
    return str_

def ms2date(ms, fmt='%Y-%m-%d'):
    if isinstance(ms, pd.Timestamp):
        return ms.strftime(fmt)
    else:
        return datetime.fromtimestamp(ms/1000).strftime(fmt)
    
def ts():
    return pd.Timestamp.now()    

def utcFromStr(tstr):
    ts = int(time.mktime(time.strptime(tstr,'%Y-%m-%d %H:%M:%S' )))
    return ts

def utcFromStrZ(tstr):
    ts = int(time.mktime(time.strptime(str(tstr),'%Y-%m-%d %H:%M:%S%z' )))
    return ts

g_order_id = int(time.time())
def getNextValidId():
    global g_order_id
    g_order_id = g_order_id + 1
    return 'smin-{}'.format(g_order_id)

def currentTime():
    return int(time.time())

def main():
    str_ = '2021-02-22 10:15:00-05:00'
    print(utcFromStrZ(str_))

if __name__ == '__main__':
    main()
