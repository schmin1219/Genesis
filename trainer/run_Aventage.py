'''
Created on Dec 25, 2020

@author: smin
'''
# from alpha_vantage.timeseries import TimeSeries
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

from IPython.display import display

#### ========================================================================
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self) 

def main():
    app = IBapi()
    app.connect('127.0.0.1', 7497, 123)
    app.run()
    time.sleep(2)
    app.disconnect()

# def main2():
#     ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
#     data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
#     
#     print (meta_data)
#     print (data.columns.tolist())
#     
#     display(data)
#     
#     data['close'].plot()
#     plt.title('Intraday Times Series for the MSFT stock (1 min)')
#     plt.show()



if __name__ == '__main__':
    main()