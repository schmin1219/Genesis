'''
Created on Jan 7, 2021

@author: smin
'''
import argparse
from common.InstMaster import InstMaster
from IPython.display import display

def updateBuy(pname, ticker, qty, price, date, cprice):
    instMaster = InstMaster()
    portfolio_json = instMaster.getInstrumentJson()[pname]
    
    ticker_idx = 0
    for idx, s in enumerate(portfolio_json):
        if s['symbol'] == ticker:
            ticker_idx = idx
            break
    
    t_json = portfolio_json[ticker_idx]
#     portfolio_json[ticker_idx]['pnl'] = t_json['pnl'] + (cprice - price)* qty if 'pnl' in t_json else (cprice - price)* qty  
    portfolio_json[ticker_idx]['pos'] = t_json['pos'] + qty if 'pos' in t_json else qty
    orders = t_json['orders'] if 'orders' in t_json else []
    o = { 'price':price, 'qty':qty, 'date':date }
    orders .append(o)
    portfolio_json[ticker_idx]['orders'] = orders
    instMaster.writePortfolioJson(pname, portfolio_json)


def updateSell(pname, ticker, qty, price, date, cprice):
    instMaster = InstMaster()
    portfolio_json = instMaster.getInstrumentJson()[pname]

    ticker_idx = 0
    for idx, s in enumerate(portfolio_json):
        if s['symbol'] == ticker:
            ticker_idx = idx
            break
    
    t_json = portfolio_json[ticker_idx]
#     orders = sorted(t_json['orders'], key=lambda k:k['price'], reverse=True) # sort worst
    orders = sorted(t_json['orders'], key=lambda k:k['price'])  # sort best 
    portfolio_json[ticker_idx]['pos'] = t_json['pos'] - qty
    for idx, o in enumerate(orders):
        portfolio_json[ticker_idx]['pnl'] = t_json['pnl'] + (price - o['price'])* o['qty'] if 'pnl' in t_json else (price - o['price'])* o['qty'] 
        qty = qty - o['qty']
        orders[idx]['qty'] = 0
        if qty == 0:
            break
        
    portfolio_json[ticker_idx]['orders'] = [ o for o in orders if o['qty'] > 0 ]
    instMaster.writePortfolioJson(pname, portfolio_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--pname', help='protfolio name', default='smin0')
    parser.add_argument('-t', '--ticker', help='068270', required=True)
    parser.add_argument('-b', '--buySell', help='B or S', required=True)
    parser.add_argument('-q', '--qty', help='[1~]', required=True)
    parser.add_argument('-p', '--price', help='price', required=True)    
    parser.add_argument('-d', '--date', help='date', required=True)
    args = parser.parse_args()
    
    if args.buySell == 'B':
        updateBuy(args.pname, args.ticker, int(args.qty), int(args.price), args.date, int(args.price))
    else:
        updateSell(args.pname, args.ticker, int(args.qty), int(args.price), args.date, int(args.price))


if __name__ == '__main__':
    main()