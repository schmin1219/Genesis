'''
Created on Dec 26, 2020

@author: smin
'''
import os
from common.readwrite import ReadFile, WriteFile

class InstMaster(object):

    def __init__(self, type, name):
        self.repo = '/home/smin/eclipse-workspace/BuySideStgy/'
#         self.inst_fp = os.path.join(self.repo, 'common', 'instruments.json')
        self.inst_fp = os.path.join(self.repo, 'common', '{}.json'.format(name))
        self.type = type
    
    def getInstrumentJson(self):
        self.inst_json = ReadFile.read_json(self.inst_fp)
        return self.inst_json
    
    def getInstNames(self):
        symbolNames= []
        for inst in self.getInstrumentJson()['instruments']:
            symbolNames.append(inst['canonical_name'])
        return symbolNames
    
    def getTickerList(self, portfolio):
        symbolNames= []
        for inst in self.getInstrumentJson()[portfolio]:
            symbolNames.append(inst['canonical_name'])
        return symbolNames
    
    def getTickSize(self, price):
        tick_size = 1000
        if price < 1000:
            tick_size = 1
        elif price >= 1000 and price < 5000:
            tick_size = 5
        elif price >= 5000 and price < 10000:
            tick_size = 10
        elif price >= 10000 and price < 50000:
            tick_size = 50
        elif price  >= 50000 and price < 100000:
            tick_size = 100
        elif price >= 100000 and price < 500000:
            tick_size = 500
        else:
            tick_size = 1000
        return tick_size
    
    def writePortfolioJson(self, pname, portfolio_json):
        self.inst_json = ReadFile.read_json(self.inst_fp)
        self.inst_json[pname] = portfolio_json
        WriteFile.write_json(self.inst_fp, self.inst_json)