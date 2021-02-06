'''
Created on Dec 26, 2020

@author: smin
'''
import os
import datetime 
import argparse
import numpy as np
import pandas as pd
from common.commondir import CommonDir
from common.InstMaster import InstMaster
from fastsim.FastSim import BuySideStgySim
from fastsim.BuySideStgySim2 import BuySideStgySim2

from IPython.display import display



class Simulator(object):

    def __init__(self, data_df, config_fp, begin_date, end_date, symbol, tradingPeriod, sim_type):
        self.tradingPeriod = tradingPeriod
        self.df  = data_df[ (data_df['Date'] >= begin_date) & (data_df['Date'] <= end_date) ]
        self.dfs = self.dividePerPeriod(begin_date, end_date)
        self.symbol = symbol
        self.sim_type = sim_type
        self.risks = { 'max_pos': 100,
                      'order_size': 1,
                      'max_loss': 10000000
                      }
        self.config = pd.read_csv(config_fp)
        self.edges = {
            'c_edge': self.config['c_edge'].iloc[0]
            ,'s_edge': self.config['s_edge'].iloc[0]
            ,'f_edge':self.config['f_edge'].iloc[0]
            }
    
    def dividePerPeriod(self, begin_date, end_date):
        periods = []
        endDate =  datetime.datetime.strptime(end_date, '%Y-%m-%d')
        _begin_date = begin_date
        _end_date = end_date
        for _ in range(1000):
            bdate = datetime.datetime.strptime(_begin_date, '%Y-%m-%d') 
            edate = bdate + datetime.timedelta(days=self.tradingPeriod)
            if endDate > edate:
                periods.append([bdate, edate])
                _begin_date = edate.strftime('%Y-%m-%d')
            else:
                periods.append([bdate, endDate])
                break
        
        
        rdf = []
        for period in periods:
            begin_date = period[0].strftime('%Y-%m-%d')
            end_date = period[1].strftime('%Y-%m-%d')
            df = self.df[ (self.df['Date'] >= begin_date) & (self.df['Date'] <= end_date) ]
            rdf.append(df)
        return rdf
    
        
    def summary(self, dfs):
        rdf = pd.DataFrame(columns=dfs.columns.tolist())
        ticker_id = dfs['ticker_id'].iloc[0]
        dfs = dfs.dropna(subset=['ave_pnl'], how='any')
        if dfs.shape[0] == 0:
            rdf.loc[0]=[ticker_id, np.NAN, np.NAN,np.NAN,np.NAN,np.NAN,0,0,0,0,0, np.NAN,np.NAN, 0, 0]
        else:
            pnls = dfs['cum_pnl'].tolist()
            apnl = np.nanmean(pnls)
            pnlsum = sum(pnls)
            pnl_std = np.nanstd(pnls)
            wpnls = [pnl for pnl in pnls if pnl > 0]
            lpnls = [pnl for pnl in pnls if pnl < 0]
            wpnl = np.nanmean(wpnls) if len(wpnls) > 0 else 0
            lpnl = np.nanmean(lpnls) if len(lpnls) > 0 else 0
            days = len(wpnls) + len(lpnls)
            sharpe = apnl / pnl_std * np.sqrt(252/self.tradingPeriod) if pnl_std > 0 else 0
            winrate = float(len(wpnls))/ float (days) if days > 0 else 0
            rdf.loc[0]=[ticker_id , apnl, wpnl, lpnl, sharpe, winrate, dfs['tot_qty'].mean()
                        ,dfs['s_edge'].iloc[0],dfs['f_edge'].iloc[0],dfs['c_edge'].iloc[0],dfs['max_pos'].max(), dfs['worst_dd'].min(), dfs['max_unreal_pnl'].max()
                        ,dfs['zc'].mean(), pnlsum ]
        return rdf
            
                    
    def run(self):
        ticker_id = self.symbol
        stgy_params_df = pd.DataFrame(columns=['ticker_id']+list(self.edges.keys()))
        stgy_params_df.loc[0] = [ticker_id] + list(self.edges.values())
        rdfs =pd.DataFrame()         
        for ddf in self.dfs:
            sim = BuySideStgySim2(ddf, self.edges['s_edge'], self.edges['f_edge'], self.edges['c_edge'], ticker_id, ticker_id, self.risks['max_pos'], self.risks['order_size'], self.risks['max_loss'])
            if self.sim_type == 0:
                sim = BuySideStgySim(ddf, self.edges['s_edge'], self.edges['f_edge'], self.edges['c_edge'], ticker_id, ticker_id, self.risks['max_pos'], self.risks['order_size'], self.risks['max_loss'])            
            sim.run()
            rdf = sim.get_res() 
            if rdf.shape[0] > 0 and ddf.shape[0] > 0:
                begin_date = ddf['Date'].iloc[0].replace('-','')
                end_date = ddf['Date'].iloc[-1].replace('-','')
                rdfs = rdfs.append(rdf)
                rdf.to_csv(os.path.join(CommonDir.sim_output_dir, 'sim_{}_{}_{}.csv'.format(ticker_id, begin_date, end_date)))
                
        rdfs = rdfs.reset_index(drop=True)
        summary_df = self.summary(rdfs)
        return summary_df, rdfs

def summary_portfolio(df, portfolio, period):
    count = 0
    pnls = []
    for _, g in df.groupby(df.index):
        pnls.append(g['cum_pnl'].sum())
        count = count + 1
    
    cum_pnl = np.sum(pnls)
    apnl = np.nanmean(pnls) if len(pnls) > 0 else np.NAN
    wpnls = [pnl for pnl in pnls if pnl > 0]
    lpnls = [pnl for pnl in pnls if pnl < 0]
    wpnl = np.nanmean(wpnls) if len(wpnls) > 0 else 0
    lpnl = np.nanmean(lpnls) if len(lpnls) > 0 else 0
    pnl_std = np.nanstd(pnls)
    sharpe = apnl/pnl_std if pnl_std > 0 else np.NAN
    sharpe = sharpe * np.sqrt(252.0/period)
    winrate = float(len(wpnls))/float(len(pnls)) if len(pnls) > 0 else 0
    rdf = pd.DataFrame(columns=['port_name', 'sharpe', 'ave_pnl', 'cum_pnl', 'win_rate', 'pos_ave_pnl', 'neg_ave_pnl', 'count'])
    rdf.loc[0] = [portfolio, sharpe, apnl, cum_pnl, winrate, wpnl, lpnl, count]
    return rdf
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--tradingPeriod', help='tradingPeriod', default=126)
    parser.add_argument('-t', '--sim_type', default=1)
    parser.add_argument('-b', '--begin_date', default='2010-01-01')
    parser.add_argument('-e', '--end_date', default='2020-12-01')
    parser.add_argument('-j', '--pname', default='instruments')
    parser.add_argument('--tag', default='SM0_21')
    args = parser.parse_args()
    
    sim_output = CommonDir.sim_output_dir
    portfolio_config_dir = CommonDir.train2_output_dir
    trading_period = int(args.tradingPeriod)
    begin_date  = args.begin_date
    end_date    = args.end_date
    sim_type = int(args.sim_type)
    prtf_name = args.pname
   
    tickers = InstMaster().getTickerList(prtf_name)
    result_dfs = pd.DataFrame()
    result_detail_df= pd.DataFrame()
    for train_dir in tickers:
        config_fp = os.path.join(portfolio_config_dir, args.tag, train_dir, 'optimize', 'final', 'strategy.csv')
        if os.path.exists(config_fp) == False:
            continue
        
        symbol = train_dir
        data_frame_df = pd.read_csv(os.path.join(CommonDir.sampled_dir,'{}.csv'.format(symbol)))
        sim = Simulator(data_frame_df, config_fp, begin_date, end_date, symbol, trading_period, sim_type)
        result_df, rdfs  = sim.run()
        
        if result_df.shape[0] > 0 :
            result_dfs  = result_dfs .append(result_df)            
            result_detail_df = result_detail_df.append(rdfs)
        else:
            print ('{} has no result '.format(train_dir))
    
    summary_df = summary_portfolio(result_detail_df, prtf_name, trading_period)
    summary_df.to_csv(os.path.join(sim_output, 'summary_{}_{}_{}.csv'.format(prtf_name, begin_date.replace('-',''), end_date.replace('-',''))))
    result_dfs.to_csv(os.path.join(sim_output, 'sim_{}_{}_{}.csv'.format(prtf_name, begin_date.replace('-',''), end_date.replace('-',''))))

    display(summary_df)    
    display(result_dfs)



if __name__ == '__main__':
    main()