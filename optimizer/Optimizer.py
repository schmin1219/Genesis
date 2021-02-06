'''
Created on Dec 2, 2020

@author: smin
'''
import os
import sys
import copy
import math
import multiprocessing
import numpy as np
import pandas as pd
from optimizer.EVOptimizer import EVOptimizer
from optimizer.SimOptimObjFunc import obj_functions
from fastsim.BuySideStgySim3 import BuySideStgySim3
from common.commondir import jsonToname
from common.commondir import CommonDir
from features.calc import FeatureClass
from IPython.display import display
from dask.dataframe.io.demo import names
from _datetime import date
pd.set_option('display.max_columns', None)
 
def calcSignals(df, signal_json, mid_cn):
    alphs_engine = FeatureClass(df, mid_cn)
    for sig in signal_json:
        cls = sig['alpha']
        eval('alphs_engine.{}(sig)'.format(cls))
    _, df = alphs_engine.results()
    return df


def runSimulation(args):
    newWeights      = args[0]
    data_dir        = args[1]
    output_dir      = args[2]
    tradingPeriod   = args[3]
    symbol          = args[4]
    sim_type        = args[5]
    date            = args[6]
    obj_func        = args[7]
    config_json     = args[8]
    signal_names    = args[9]
        
    max_pos         = config_json['strategy']['max_pos']
    order_size      = config_json['strategy']['order_size']
    max_loss        = config_json['strategy']['max_loss']
    ddfp = os.path.join(data_dir, '{}.{}.csv.gz'.format(symbol, date))
    if os.path.exists(ddfp)==False:
        return date, pd.DataFrame()
    
    ddf = pd.read_csv(ddfp, compression = 'gzip')
    mid_cn = [ cn for cn in ddf.columns.tolist() if cn.startswith('MidPriceSignal') ][0]
    ddf = calcSignals(ddf, config_json['bison_signals'], mid_cn)
    rdfs = pd.DataFrame()
    
    sedge_init = config_json['salmon_signals']['s_edge']
    fedge_init = config_json['salmon_signals']['f_edge']
    cedge_init = config_json['salmon_signals']['c_edge']
    bedge_init = config_json['salmon_signals']['b_edge']
    for idx in range(0, len(newWeights)):
        weights = newWeights[idx]
        sedge = weights[0] * sedge_init
        fedge = weights[1] * fedge_init
        cedge = weights[2] * cedge_init
        bedge = weights[3] * bedge_init
        
#         #TODO ----------------------------
#         print (weights[0],sedge_init, sedge)
#         print (weights[1],fedge_init, fedge)
#         print (weights[2],cedge_init, cedge)
#         print (weights[3],bedge_init, bedge)
#         
        ddf['alpha'] = (ddf[signal_names] * weights[4:]).sum(axis=1) * ddf[mid_cn]
        sim = BuySideStgySim3(ddf, sedge, fedge, cedge, bedge, symbol, idx, max_pos, order_size,max_loss)
        sim.run()
        rdf = sim.get_res() 
        if rdf.shape[0] > 0:
            rdf['ticker_id'] = idx
            rdfs = rdfs.append(rdf)
    return date, rdfs


class Optimizer(object):

    def __init__(self, data_dir, output_dir, tradingPeriod, symbol, sim_type, begin_date, end_date, obj_func_, json_, intra_trading):
        self.json_ = json_
        self.tradingPeriod = tradingPeriod
        self.intra_trading = intra_trading
        self.data_dir = data_dir
        self.sim_type = sim_type
        self.output_dir = output_dir
        self.obj_func = obj_func_
        self.optimization_id = 'optimize'            
        self.risks = { 'max_pos'  : int(json_['strategy']['max_pos']),
                      'order_size': int(json_['strategy']['order_size']),
                      'max_loss'  : int(json_['strategy']['max_loss']),
                      }
        
        self.symbol = symbol
        self.edges = {
            'c_edge' : json_['salmon_signals']['c_edge']
            ,'s_edge': json_['salmon_signals']['s_edge']
            ,'f_edge': json_['salmon_signals']['f_edge']
            ,'b_edge': json_['salmon_signals']['b_edge']
            }

        self.signals_names = [ jsonToname(s) for s in json_['bison_signals'] ]
        self.signals = json_['bison_signals']
        self.optim = EVOptimizer(self.edges, self.signals_names, json_)
        self.tradingDays = self.getTradingDays(begin_date, end_date)
        print ('INFO, FastSimOptimizer::init() numStrategies, FastSimOptimizer') 
        

    def getTradingDays(self, bdate, edate):
        dates = []
        bdate_ = int(bdate.replace('-',''))
        edate_ = int(edate.replace('-',''))
        print ('{} ~ {}'.format(bdate, edate))
        
        for fn in os.listdir(self.data_dir):
            if fn.startswith('{}'.format(self.symbol)) and fn.endswith('csv.gz'):
                d_ = int(fn.split('.')[1])
                dates.append(d_)
                
        dates = [ d for d in dates if d >= int(bdate_) and d <= int(edate_)]
        dates = sorted(dates)
        return dates

    def summary(self,newWeights, dfs):
        names = ['ticker_id', 'ave_pnl', 'win_pnl', 'loss_pnl', 'sharpe', 'win_rate', 'tot_qty', 's_edge', 'f_edge', 'c_edge', 'b_edge', 'max_pos', 'worst_dd', 'max_unreal_pnl', 'zc', 'cum_pnl'] + self.signals_names
        score_df = pd.DataFrame(columns=names)
        for _, id_df in dfs.groupby('ticker_id'):
            idx = int(id_df['ticker_id'].iloc[0])
            weights = newWeights[idx]
            
            pnls = [ pnl for pnl in id_df['ave_pnl'].tolist() if pnl != 0 ]
            wpnls = [ pnl for pnl in id_df['ave_pnl'].tolist() if pnl > 0 ]
            lpnls = [ pnl for pnl in id_df['ave_pnl'].tolist() if pnl < 0 ]
            apnl = np.nanmean(pnls)
            wpnl = np.nanmean(wpnls) if len(wpnls) > 0 else 0
            lpnl = np.nanmean(lpnls) if len(lpnls) > 0 else 0
            sharpe = apnl / np.nanstd(pnls) * np.sqrt(252) if np.nanstd(pnls) > 0 else np.NAN
            days = len(wpnls) + len(lpnls)
            wr = float(len(wpnls))/ float (days) if days > 0 else 0
            cumpnl = np.nansum(pnls)
            zc = id_df['zc'].mean()
            max_pos = id_df['max_pos'].max()
            wdd = id_df['worst_dd'].min()
            maxdd= id_df['max_unreal_pnl'].max()
            qty = id_df['tot_qty'].mean()
            ticker_id =id_df['ticker_id'].iloc[0]
            score_df.loc[score_df.shape[0]] = [ticker_id,apnl,wpnl,lpnl,sharpe,wr,qty,weights[0],weights[1],weights[2],weights[3],max_pos,wdd,maxdd,zc,cumpnl]+list(weights[4:])
        score_df = score_df.reset_index(drop=True)
        return score_df
        
                        
    #  Should do multiple simulation of multiple period 
    def runParallelSimulation(self, newWeights):
        cnt = multiprocessing.cpu_count() - 3
        pool = multiprocessing.Pool(processes=cnt)
        process_args = [ 
                        [newWeights,
                         self.data_dir,
                         os.path.join(self.output_dir, '{}'.format(self.symbol)),
                         self.tradingPeriod,
                         self.symbol,
                         self.sim_type, 
                         date,
                         self.json_['obj_func'],
                         copy.deepcopy(self.json_),
                         self.signals_names
                      ]  for date in self.tradingDays ]    
        
        process_result = pool.map(runSimulation, process_args)
        pool.terminate()    
        
        res = pd.DataFrame()
        for result in process_result:
            if result[1].shape[0] > 0:
                res = res.append(result[1])
        
        score_df = self.summary(newWeights, res)
        return score_df
        
    def optimize(self):
        optimize_root_dir = os.path.join(self.output_dir, self.optimization_id)
        if not os.path.exists(optimize_root_dir):
            os.makedirs(optimize_root_dir)
            os.system('chmod -R 777 {}'.format(self.output_dir))

        historical_oos_df = pd.DataFrame()
        historical_df     = pd.DataFrame()
        best_score_df_ovall = pd.DataFrame()
        
        
        for i in range(0, self.optim.maxloop()):
            gen_root_dir = os.path.join(optimize_root_dir, str(i)+'_generation')
            if not os.path.exists(gen_root_dir):
                os.makedirs(gen_root_dir)
            newAsks = self.optim.ask(True if i == 0 else False, 0.05)
            result_df = self.runParallelSimulation(newAsks)
            score_df, isMax = self.calculateScoring(result_df, obj_functions[self.obj_func])

            if score_df.empty:
                if self.optim.maxloop() != i+1:
                    print ('score_df is empty {}'.format(i))
                    continue
                else:
                    break
            
            score_df['ticker'] = self.symbol
            score_stgy_df = score_df
            historical_df = historical_df.append(score_stgy_df)
            score_stgy_df['score'] = score_stgy_df.score
            best_score_idx = float('nan') if score_stgy_df.shape[0] == 0 else score_stgy_df[['score']].idxmax()
            
            if math.isnan(best_score_idx):
                if self.optim.maxloop() != i+1:
                    continue
                else:
                    self.__finalstage(score_stgy_df, best_score_df_ovall,historical_df,gen_root_dir,optimize_root_dir,isMax,historical_oos_df)
                    break
            
            best_score_df_ovall = best_score_df_ovall.append(score_stgy_df.iloc[[int(best_score_idx)]], ignore_index=True, sort=False)
            self.optim.tell(newAsks, list(score_stgy_df['score']), isMax)
            if self.optim.exitCondition() == True or self.optim.maxloop() == i+1:
                self.__finalstage(score_stgy_df, best_score_df_ovall,historical_df,gen_root_dir,optimize_root_dir,isMax,historical_oos_df)                
                break
            score_stgy_df.to_csv(os.path.join(gen_root_dir, 'score_stgy.csv'))
            best_score_df_ovall.to_csv(os.path.join(gen_root_dir, 'best_score_overall.csv'))
            print ('{} Loop={}, '.format(self.symbol, i))
            display(best_score_df_ovall.tail(1))
             
                
    def __finalstage(self, score_stgy_df, best_score_df_ovall,historical_df,gen_root_dir,optimize_root_dir,isMax,historical_oos_df):
        score_stgy_df.to_csv(os.path.join(gen_root_dir, 'score_stgy.csv'))
        best_score_df_ovall.to_csv(os.path.join(gen_root_dir, 'best_score_overall.csv'))
        self.final_output_dir = os.path.join(optimize_root_dir, 'final')
        if not os.path.exists(self.final_output_dir):
            os.makedirs(self.final_output_dir)
        
        if best_score_df_ovall.shape[0] > 0 :
            best_score_df = best_score_df_ovall.sort_values(by=['score']).tail(1)
            best_score_df.to_csv(os.path.join(self.final_output_dir, 'strategy.csv'))
            best_score_df_ovall.to_csv(os.path.join(self.final_output_dir, 'best_score_overall.csv'))
            historical_df.to_csv(os.path.join(self.final_output_dir, 'historical_optimization.csv'))
            display(best_score_df_ovall)

    def calculateScoring(self, df, obj_func):
        if df.empty:
            print >> sys.stderr, "WARN, simulations don't have results, calculateScoring"
            return df, True 
    
        df['ticker_id'] = df['ticker_id'].apply(str)
        df_stgy_list = df.groupby(['ticker_id'])
        stgy_sum_df = pd.DataFrame()
        for _, df_stgy in df_stgy_list:
            _, score, isMax = obj_func(df_stgy)
            df_stgy['score'] = score
            stgy_sum_df = stgy_sum_df.append(df_stgy, ignore_index=True, sort=False)
        return stgy_sum_df, isMax

 
