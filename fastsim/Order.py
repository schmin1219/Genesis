'''
Created on Dec 1, 2020

@author: smin
'''
from datetime import date

class MyOrder(object):
    def __init__(self, orderId, isBuy, price, qty, day):
        self.order_qty = qty
        self.price = price
        self.isBuy = isBuy
        self.orderId = orderId
        self.date = day

    def convDate(self, day):
        tokens = day.split('-')
        return date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
    
    def expired(self, today):
        order_date = self.convDate(self.date)
        today_date = self.convDate(today)
        return abs(today_date - order_date).days
        

                
#### ========================================================================
def main():
    print ('Test')
    ord = MyOrder(1, True, 100, 1, 20201103)
    distance = ord.expired(20201209)
    print (distance)


if __name__ == '__main__':
    main()