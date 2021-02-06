'''
Created on Jan 29, 2021

@author: smin
'''
import pandas as pd
import numpy as np
from common.commondir import jsonToname

class FeatureClass(object):

    def __init__(self, data, mid_cn):
        '''
        Constructor
        '''
        self.data_df = data
        self.mid_cn = mid_cn
        self.feature_names = []
    
    
    def ema_impact(self, signal_json):
        cn = jsonToname(signal_json)
        lag = signal_json['config']['lag']
        ema = pd.Series.ewm( self.data_df[self.mid_cn], halflife = lag/2.0).mean()
        self.data_df[cn] = (ema - self.data_df[self.mid_cn])/self.data_df[self.mid_cn]
        self.feature_names.append(cn) 

    def sma_impact(self, signal_json):
        cn = jsonToname(signal_json)
        lag = signal_json['config']['lag']
        sma = self.data_df[self.mid_cn].rolling(lag).mean()
        self.data_df[cn] = (sma - self.data_df[self.mid_cn])/self.data_df[self.mid_cn]
        self.feature_names.append(cn) 

    #TODO 
    def tt_impact(self, signal_json):
        '''
            upimpact = buyvwap - midprice 
            upimpact = upimpact if upimpact > 0 else 0
             
            dnimpact = midprice - soldvwap 
            dnimpact = dnimpact if dnimpact > 0 else 0
            
            # We can add more conditional concept !! 
            
            edge = (upimpact - dnimpact)/df[mid_cn]
        '''
        pass

    
    def results(self):
        return self.feature_names, self.data_df
    