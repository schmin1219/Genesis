'''
Created on Dec 27, 2020

@author: smin
'''
import json

class CommonDir(object):

    def __init__(self):
        pass

    repo_dir = '/home/smin/eclipse-workspace/BuySideStgy/'
    train_output_dir = '/home/smin/StockEngine/results/'
    train2_output_dir = '/home/smin/StockEngine/results/'
    sim_output_dir = '/home/smin/StockEngine/sim/'
    sampled_dir = '/home/smin/StockEngine/data/'
    sampled_us_dir = '/home/smin/StockEngine/us_data/'
    portfolio_dir = '/home/smin/StockEngine/portfolio/'
    trading_dir = '/home/smin/StockEngine/trading/'
    
    


def jsonToname(json_):
    str_ = json.dumps(json_)
    str_ = str_.replace('"', '').replace('{','').replace('}','').replace(':','').replace(' ','').strip()
    return str_