import os
import re
import pytz
import json
import copy
import math
import pandas as pd
import numpy as np
import timeit
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from IPython.display import display
import multiprocessing
from itertools import combinations
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows',500)


def read_json(jpath):
    with open(jpath) as f:
        reg_obj = json.load(f)
    return reg_obj

def save_json(j, fpath):
    with open(fpath, 'w') as f:
        f.write(json.dumps(j, indent=4, sort_keys=True))


def portfolio_cal(stgylist):
    intra_min_pnl = {}
    intra_max_pnl = {}
    pnl_map = {}
    
    candidates = ','.join(stgylist)
    stgylistdir = [ os.path.dirname(_dir) for _dir in stgylist ]
    for sfn in stgylistdir:
        for _fn in os.listdir(sfn):
            if _fn.startswith('packresults'):
                _df = pd.read_csv(os.path.join(sfn, _fn))
                pnl = 0
                if _df.shape[0] > 0:
                    worst_intraday_pnl  = float(_df['minRealizedPNL'].min())
                    best_intraday_pnl = float(_df['maxRealizedPNL'].max())
                    pnl = float(_df['eqoPNLValue'].mean(skipna=True))
                    if pnl != 0:                
                        d = int(_fn.split('.')[1])
                        if d not in pnl_map:
                            pnl_map[d] = pnl
                            intra_max_pnl[d] = best_intraday_pnl
                            intra_min_pnl[d] = worst_intraday_pnl
                        else:
                            pnl_map[d] = pnl + pnl_map[d]
                            intra_max_pnl[d] = best_intraday_pnl + intra_max_pnl[d]
                            intra_min_pnl[d] = worst_intraday_pnl + intra_min_pnl[d]
 
    data = {
            'date': list(pnl_map.keys())
            , 'eqoPNLValue': list(pnl_map.values())
            , 'minRealizedPNL': list(intra_min_pnl.values())
            , 'maxRealizedPNL': list(intra_max_pnl.values())
           } 
    df = pd.DataFrame(data, columns=['date','eqoPNLValue', 'minRealizedPNL', 'maxRealizedPNL'])                    
 
    average_pnl =df['eqoPNLValue'].mean()
    stdev_pnl  = df['eqoPNLValue'].std()
    sharpe = average_pnl / stdev_pnl * math.sqrt(252)
    winrate = float(df [ df['eqoPNLValue'] > 0 ].shape[0]) / float(len(pnl_map))
    
    column_names = ['annualSharpe','days','eqoPNLValue', 'worstDayPnL', 'bestDayPnL', 'worstItraPnL', 'bestIntraPnL', 'WinRate','candidates' ]
    port_data = [sharpe, len(pnl_map), average_pnl, df['eqoPNLValue'].min(), df['eqoPNLValue'].max()
                 , df['minRealizedPNL'].min(), df['minRealizedPNL'].max()
                 , winrate, candidates
                 ]
    port_df = pd.DataFrame(columns=[column_names])
    port_df.loc[len(port_df)] = port_data
    df = df.sort_values(by=['date'])
    return df, port_df

def portfolio_cal2(argument):
    stgylist = argument[0]
    intra_min_pnl = {}
    intra_max_pnl = {}
    pnl_map = {}
    
    candidates = ','.join(stgylist)
    stgylistdir = [ os.path.dirname(_dir) for _dir in stgylist ]
    for sfn in stgylistdir:
        for _fn in os.listdir(sfn):
            if _fn.startswith('packresults'):
                _df = pd.read_csv(os.path.join(sfn, _fn))
                pnl = 0
                if _df.shape[0] > 0:
                    worst_intraday_pnl  = float(_df['minRealizedPNL'].min())
                    best_intraday_pnl = float(_df['maxRealizedPNL'].max())
                    pnl = float(_df['eqoPNLValue'].mean(skipna=True))
                    if pnl != 0:                
                        d = int(_fn.split('.')[1])
                        if d not in pnl_map:
                            pnl_map[d] = pnl
                            intra_max_pnl[d] = best_intraday_pnl
                            intra_min_pnl[d] = worst_intraday_pnl
                        else:
                            pnl_map[d] = pnl + pnl_map[d]
                            intra_max_pnl[d] = best_intraday_pnl + intra_max_pnl[d]
                            intra_min_pnl[d] = worst_intraday_pnl + intra_min_pnl[d]
 
    data = {'date': list(pnl_map.keys())
            , 'eqoPNLValue': list(pnl_map.values())
            , 'minRealizedPNL': list(intra_min_pnl.values())
            , 'maxRealizedPNL': list(intra_max_pnl.values())
           } 
    df = pd.DataFrame(data, columns=['date','eqoPNLValue', 'minRealizedPNL', 'maxRealizedPNL'])                    
 
    average_pnl =df['eqoPNLValue'].mean()
    stdev_pnl  = df['eqoPNLValue'].std()
    sharpe = average_pnl / stdev_pnl * math.sqrt(252)
    winrate = float(df [ df['eqoPNLValue'] > 0 ].shape[0]) / float(len(pnl_map))
    
    column_names = ['annualSharpe','days','eqoPNLValue', 'worstDayPnL', 'bestDayPnL', 'worstItraPnL', 'bestIntraPnL', 'WinRate','candidates' ]
    port_data = [sharpe, len(pnl_map), average_pnl, df['eqoPNLValue'].min(), df['eqoPNLValue'].max()
                 , df['minRealizedPNL'].min(), df['minRealizedPNL'].max()
                 , winrate, candidates
                 ]
    port_df = pd.DataFrame(columns=[column_names])
    port_df.loc[len(port_df)] = port_data
    df = df.sort_values(by=['date'])
    return df, port_df

def getJsonAndPutInProdDir(target_dir, prod):
    fn = ''
    for _json_fn in os.listdir(target_dir):
        if _json_fn.endswith('.json'):
            fn = os.path.join(target_dir, _json_fn)
            break
    
    if fn and os.path.exists(fn):
        _json = read_json(fn)
        uid = os.path.basename(fn).split('.')[-2]
        print (uid)
        output_dir = '/home/smin/smin/output/portfolio/{}/'.format(prod)
        json_fn = 'strategy.train_TRAINER_{}_0_{}_{}_Z{}_0.json'.format(prod,_json['begin_date'], _json['end_date'],uid)
        save_json(_json, os.path.join(output_dir, json_fn))
        
def search_candidates_set(candidate_list, num_stgy):
    candidates = []
    for c in sum([map(list, combinations(candidate_list, i)) for i in range(len(candidate_list) + 1)], []):
        if len(c) == num_stgy:
            candidates.append(c)
    
    process_args = []
    for c in candidates:
        process_args.append([c])
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 5)
    process_result = pool.map(portfolio_cal2, process_args)
    pool.terminate()

    res = pd.DataFrame()
    for result in process_result:
        if result[1].shape[0] > 0:
            res = res.append(result[1])
    return res

def run_os(args_dir, args_winrate, args_sharpe, args_days, args_oos_sharpe):
    candidate_df = pd.DataFrame()
    for port_dir in os.listdir(args_dir):
        if os.path.isfile(os.path.join(args_dir, port_dir)):
            continue

        summary_packresults_fn = ''
        candidate_json_fn =''
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.startswith('is_summary_packresults'):
                summary_packresults_fn = os.path.join(args_dir, port_dir, fn)
                break
        if not summary_packresults_fn:
            continue

        summary_packresults_df = pd.read_csv(summary_packresults_fn)
        summary_packresults_df = summary_packresults_df[ summary_packresults_df['days'] > 1]
        summary_packresults_df['comment'] = summary_packresults_fn
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.endswith('.json'):
                candidate_json_fn = os.path.join(args_dir, port_dir, fn)
                break
        if not candidate_json_fn:
            candidate_json_fn = os.path.join(args_dir, port_dir)

        oos_summary_packresults_fn = ''
        is_oos = False
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.startswith('oos_summary_packresults'):
                oos_summary_packresults_fn = os.path.join(args_dir, port_dir, fn)
                is_oos = True
                break

        if is_oos:
            oos_summary_packresults_df = pd.read_csv(oos_summary_packresults_fn)
        summary_packresults_df['oos_annualSharpe'] =  oos_summary_packresults_df.iloc[0, oos_summary_packresults_df.columns.get_loc('annualSharpe')] if is_oos else 0
        summary_packresults_df['oos_days'] =  oos_summary_packresults_df.iloc[0, oos_summary_packresults_df.columns.get_loc('days')] if is_oos else 0
        summary_packresults_df['oos_winrate'] =  oos_summary_packresults_df.iloc[0, oos_summary_packresults_df.columns.get_loc('winrate')] if is_oos else 0
        summary_packresults_df['oos_eqoPNLValue'] =  oos_summary_packresults_df.iloc[0, oos_summary_packresults_df.columns.get_loc('eqoPNLValue')] if is_oos else 0
        summary_packresults_df['candidate'] = candidate_json_fn
        candidate_df = candidate_df.append(summary_packresults_df)


    candidate_df = candidate_df[candidate_df['winrate'] > float(args_winrate)]
    candidate_df = candidate_df[candidate_df['annualSharpe'] > float(args_sharpe)]
    candidate_df = candidate_df[candidate_df['days'] > float(args_days)]
    candidate_df = candidate_df[candidate_df['oos_annualSharpe'] > float(args_oos_sharpe)]
    candidate_df = candidate_df.sort_values(by=['annualSharpe'], ascending=False)
    return candidate_df


def run(args_dir, args_winrate, args_sharpe, args_days):
    candidate_df = pd.DataFrame()
    for port_dir in os.listdir(args_dir):
        if os.path.isfile(os.path.join(args_dir, port_dir)):
            continue

        summary_packresults_fn = ''
        candidate_json_fn =''
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.startswith('summary_packresults'):
                summary_packresults_fn = os.path.join(args_dir, port_dir, fn)
                break
        if not summary_packresults_fn:
            continue

        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.endswith('.json'):
                candidate_json_fn = os.path.join(args_dir, port_dir, fn)
                break
            
        if not candidate_json_fn:
            candidate_json_fn = os.path.join(args_dir, port_dir)
        else:
            _json = read_json(candidate_json_fn)
        
        if summary_packresults_fn != '':
            summary_packresults_df = pd.read_csv(summary_packresults_fn)
            if summary_packresults_df.shape[0] > 0 :
                summary_packresults_df = summary_packresults_df[ summary_packresults_df['days'] > 1]
                summary_packresults_df['comment'] = candidate_json_fn
                summary_packresults_df['max_pos'] = _json['strategy']['risk']['max_pos'] if candidate_json_fn else 0
                summary_packresults_df['strategy'] = _json['strategy']['trigger_name'] if candidate_json_fn else 'NA'
                if 'profit_cover_ticks' in _json['strategy']:
                    summary_packresults_df['profit_cover_ticks'] = _json['strategy']['profit_cover_ticks'] if candidate_json_fn else 0
                else:
                    summary_packresults_df['profit_cover_ticks'] = 0
                    
        
                for fn in os.listdir(os.path.join(args_dir, port_dir)):
                    if fn.endswith('.json'):
                        candidate_json_fn = os.path.join(args_dir, port_dir, fn)
                        break
                        
                if not candidate_json_fn:
                    candidate_json_fn = os.path.join(args_dir, port_dir)
                candidate_df = candidate_df.append(summary_packresults_df)

    if candidate_df.shape[0] > 0:
        candidate_df = candidate_df[candidate_df['winrate'] > float(args_winrate)]
        candidate_df = candidate_df[candidate_df['annualSharpe'] > float(args_sharpe)]
        candidate_df = candidate_df[candidate_df['days'] > float(args_days)]
        candidate_df = candidate_df.sort_values(by=['annualSharpe'], ascending=False)
    return candidate_df


def run2(args_dir, args_winrate, args_sharpe, args_days):
    candidate_df = pd.DataFrame()
    for port_dir in os.listdir(args_dir):
        if os.path.isfile(os.path.join(args_dir, port_dir)):
            continue

        summary_packresults_fn = os.path.join(args_dir, port_dir, 'summary_packresults_20190101_20191130.csv')
        if os.path.exists(summary_packresults_fn) == False:
            continue

        candidate_json_fn =os.path.join(args_dir, port_dir, '{}.json'.format(port_dir))
        if os.path.exists(candidate_json_fn) == False:
            candidate_json_fn = os.path.join(args_dir, port_dir)
        else:
            _json = read_json(candidate_json_fn)
        
        if summary_packresults_fn != '':
            summary_packresults_df = pd.read_csv(summary_packresults_fn)
            if summary_packresults_df.shape[0] > 0 :
                summary_packresults_df = summary_packresults_df[ summary_packresults_df['days'] > 1]
                summary_packresults_df['comment'] = candidate_json_fn
                summary_packresults_df['max_pos'] = _json['strategy']['risk']['max_pos'] if len(_json) > 0  else 0
                summary_packresults_df['strategy'] = _json['strategy']['trigger_name'] if len(_json) > 0 else 'NA'
                if 'profit_cover_ticks' in _json['strategy']:
                    summary_packresults_df['profit_cover_ticks'] = _json['strategy']['profit_cover_ticks'] if len(_json) > 0 else 0
                else:
                    summary_packresults_df['profit_cover_ticks'] = 0

                candidate_df = candidate_df.append(summary_packresults_df)

    if candidate_df.shape[0] > 0:
        candidate_df = candidate_df[candidate_df['winrate'] > float(args_winrate)]
        candidate_df = candidate_df[candidate_df['annualSharpe'] > float(args_sharpe)]
        candidate_df = candidate_df[candidate_df['days'] > float(args_days)]
        candidate_df = candidate_df.sort_values(by=['annualSharpe'], ascending=False)
    return candidate_df


def runWithDay(args_dir, args_pnl, _day):
    candidate_df = pd.DataFrame()
    for port_dir in os.listdir(args_dir):
        if os.path.isfile(os.path.join(args_dir, port_dir)):
            continue

        summary_packresults_fn = ''
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.startswith('summary_packresults'):
                summary_packresults_fn = os.path.join(args_dir, port_dir, fn)
                break
        if not summary_packresults_fn:
            continue
            
        packresults_fn = ''
        candidate_json_fn =''
        for fn in os.listdir(os.path.join(args_dir, port_dir)):
            if fn.startswith('packresults.{}'.format(_day)):
                packresults_fn = os.path.join(args_dir, port_dir, fn)
                break
        if not packresults_fn:
            continue
                
        if not candidate_json_fn:
            candidate_json_fn = os.path.join(args_dir, port_dir)

        summary_packresults_df = pd.read_csv(summary_packresults_fn)
        packresults_df = pd.read_csv(packresults_fn)
        
        packresults_df['comment'] = candidate_json_fn
        packresults_df['aSharpe'] = summary_packresults_df['annualSharpe']
        packresults_df['trd_days'] = summary_packresults_df['days']
        packresults_df['totWinrate'] = summary_packresults_df['winrate']
        if not candidate_json_fn:
            candidate_json_fn = os.path.join(args_dir, port_dir)
        candidate_df = candidate_df.append(packresults_df)

    candidate_df = candidate_df[candidate_df['eqoPNLValue'] > float(args_pnl)]
    return candidate_df

def getOrderTimingsMap(stgylist):
    orderTimingsMap = {}
    for sfn in stgylist:
        for _fn in os.listdir(sfn):
            if _fn.startswith('orderTimings'):
                _df = pd.read_csv(os.path.join(sfn, _fn))
                if _df.shape[0] > 0:
                    d = int(_fn.split('.')[1])
                    _df['ts'] = _df.packetTransactTimestamp.apply(lambda d: datetime.datetime.fromtimestamp(int(d/1000000000)).strftime('%H:%M:%S'))
                    if d not in orderTimingsMap:
                        orderTimingsMap[d] = _df
                    else:
                        orderTimingsMap[d] = orderTimingsMap[d].append(_df).sort_values(by=['packetReceivedTimestamp'])
    return orderTimingsMap

def tradingSimBehavior(df, _d):
    df_plot = df.copy()
    xs = df_plot.index

    df_plot['signedFillQty'] = df_plot['side'] * df_plot['fillQty']
    df_plot['filled'] = (df_plot['fillQty'] > 0) + 0
    df_plot['trade'] = df_plot['side'] * df_plot['filled'] * df_plot['qty']
    df_plot['netPos'] = df_plot['trade'].cumsum()
    df_plot['alphaValue'] = (df_plot['alphaAtSendTime'] - df_plot['midPrice']) / (df_plot['askPriceAtSendTime'] - df_plot['bidPriceAtSendTime'])
    df_plot['prevInvestment'] = (df_plot['fillPrice'] * df_plot['signedFillQty']) 
    df_plot['cPnl'] = np.where(df_plot['filled'] == 0, 0, df_plot['netPos'] * df_plot['fillPrice'] - df_plot['prevInvestment'].fillna(0).cumsum())
    df_plot['buyFill'] = np.where(df_plot['signedFillQty'] > 0, df_plot['fillPrice'], np.NaN)
    df_plot['selFill'] = np.where(df_plot['signedFillQty'] < 0, df_plot['fillPrice'], np.NaN)
    df_plot['buyOrder'] = np.where(df['side'] > 0, df_plot['askPriceAtSendTime'], np.NaN)
    df_plot['selOrder'] = np.where(df['side'] < 0, df_plot['bidPriceAtSendTime'], np.NaN)
    df_plot['date'] = _d
    
    display(df_plot[['date','packetReceivedTimestamp','ts','fillPrice','cPnl','netPos']])
#     display(df_plot)
        
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(11, 1)
    gs.update(wspace=0, hspace=0)
    ax1 = plt.subplot(gs[0:5, :])
    ax2 = plt.subplot(gs[5:7, :])
    ax3 = plt.subplot(gs[7:9, :])
    
    ax1.set_title('{}'.format(_d))
    ax1.plot(xs, df_plot['bidPriceAtSendTime'], c='gray', drawstyle='steps-post', alpha=0.5)
    ax1.plot(xs, df_plot['askPriceAtSendTime'], c='gray', drawstyle='steps-post', alpha=0.5)
    ax1.plot(xs, df_plot['buyFill'], c='blue', marker='o', ms=10)
    ax1.plot(xs, df_plot['selFill'], c='red', marker='o', ms=10)
    ax1.plot(xs, df_plot['buyOrder'], lw=0, c='blue', marker='o', ms=10, markerfacecolor='white', markeredgecolor='blue', zorder=0, alpha=0.5)
    ax1.plot(xs, df_plot['selOrder'], lw=0, c='red', marker='o', ms=10, markerfacecolor='white', markeredgecolor='red', zorder=0, alpha=0.5)

    ax2.plot(xs, df_plot['cPnl'], drawstyle='steps-post')
    ax2.axhline(y=0, c='gray', lw=2, linestyle=':', alpha=0.5)
    ax3.plot(xs, df_plot['netPos'], drawstyle='steps-post', alpha=0.5)
    ax3.axhline(y=0, c='gray', lw=2, linestyle=':', alpha=0.5)

def day_pnl_graph(df):
    df_daily = df.sort_values(by=['date'])
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(9, 1)
    gs.update(wspace=0, hspace=0)
    ax1 = plt.subplot(gs[0:9, :])
    ax1.plot(df_daily['date'], df_daily['eqoPNLValue'].cumsum(), alpha=0.9)


def getInstMaster(api, symbol, cfg, begin_date, end_date, sampling_path):
    ref_symbols = [symbol ]
    for _signal in cfg:
        if 'symbol' in _signal['config']:
            if _signal['config']['symbol'] not in ref_symbols:
                ref_symbols.append(_signal['config']['symbol']) 
        elif 'refSymbol' in _signal['config']:
            if _signal['config']['refSymbol'] not in ref_symbols:
                ref_symbols.append(_signal['config']['refSymbol']) 

    availableDates = api.getAvailableDates(begin_date,end_date, sampling_path)
    secMap = api.secLookup(int(availableDates[-1]), ref_symbols)
    
    secMapList = []
    for _sec in secMap:
        secMapList.append(secMap[_sec])
    
    return ref_symbols, secMapList

def getRefSymbols(symbol, cfg):
    ref_symbols = [symbol ]
    for _signal in cfg:
        if 'symbol' in _signal['config']:
            if _signal['config']['symbol'] not in ref_symbols:
                ref_symbols.append(_signal['config']['symbol']) 
        elif 'refSymbol' in _signal['config']:
            if _signal['config']['refSymbol'] not in ref_symbols:
                ref_symbols.append(_signal['config']['refSymbol']) 

    return ref_symbols

  

def buildCombinedZStgySignal(trigger_name, prod, candidates, output_dir, noFilter=False):
    strategies_map = {}
    strategies_signal_config = {}
    strategies_signal_hash = {}
    pct_lst = [1,2,3,4,5,6,7,8,9,1000]
    for i, c in enumerate(candidates):
        _json = read_json(c)
        
        if noFilter:
            _stgy_hash = 888
        else:
            _stgy_hash = hash(json.dumps(_json['strategy']['profit_cover_ticks'], sort_keys=True) + json.dumps(_json['strategy']['trigger_name']))
              
        if _stgy_hash not in strategies_map:
            strategies_map[_stgy_hash] = _json
            strategies_signal_config[_stgy_hash] = []
            strategies_signal_hash[_stgy_hash] = []   
        
        for _sig in _json['signal_config']:
            _sig_hash = hash( json.dumps(_sig['class'],sort_keys=True) + json.dumps(_sig['config'], sort_keys=True))
            if _sig_hash not in strategies_signal_hash[_stgy_hash]:
                strategies_signal_hash[_stgy_hash].append(_sig_hash)
                strategies_signal_config[_stgy_hash].append(_sig)
    
    for i, _stgy in enumerate(strategies_map):
        for pct in pct_lst:
            strategies_map[_stgy]['output']['output_dir'] = '/data/eqo-t8-nas-01/eqo2/workspace/smin/results/'
            strategies_map[_stgy]['sampling_path'] = '/data/group/eqo/eqo_smin/'
            strategies_map[_stgy]['framework']['task_submitter'] = '/data/eqo-t8-nas-01/eqo2/common/ClusterRunEqoBigSizePersist.py'
            strategies_map[_stgy]['signal_config'] = strategies_signal_config[_stgy]
            strategies_map[_stgy]['signal_config_weights'] = [1.0] * len(strategies_map[_stgy]['signal_config'])
            strategies_map[_stgy]['simulator']['coeff_sets'][0]['coeffs'] = [1.0] * len(strategies_map[_stgy]['signal_config'])
            strategies_map[_stgy]['strategy']['risk']['max_pos'] = 4
            strategies_map[_stgy]['strategy']['profit_cover_ticks'] = pct
            
            output_fn = '{}_{}_{}_{}_{}.json'.format(prod, trigger_name, len(strategies_map[_stgy]['signal_config']),pct, i)
            save_json(strategies_map[_stgy], os.path.join(output_dir, output_fn))
            print ('{}, signal length = {}'.format(os.path.join(output_dir, output_fn), len(strategies_map[_stgy]['signal_config'])))





##################################### example ##########################################
def main():
    args_dir = '/home/smin/smin/results/portfolio/YM206_D'
    args_winrate = 0.0
    args_sharpe = 3
    args_days = 30
    df1 = run(args_dir, args_winrate, args_sharpe, args_days)
    # df1 = df1[ (df1.days > 70) & (df1.annualSharpe > 5) & (df1.tradedQty > 7)]
    display(df1.sort_values(by=['days', 'annualSharpe'], ascending=False))
    
    pass

if __name__ == '__main__':
    main()