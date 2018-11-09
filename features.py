import time
import numpy as np
from threading import Thread
from debug import d


    
class FeatureProcess(Thread):    
    def __init__(self, buffer,sender):
        Thread.__init__(self)
        self.buffer=buffer
        self.N, self.hs=buffer.N, buffer.hopsize
        self.sender=sender
        self.cur_data=None
        self.cur_labels=None
        self.out_data=None
        self.lT=time.time()
    def clean_data(self, data):     
        cT=time.time()
        self.cur_labels= np.array([d_k['label'] for d_k in data[0]['data']])        
        last_labels=np.array([d_k['label'] for d_k in data[-1]['data']])        
        
        if not np.all(self.cur_labels==last_labels):
            self.cur_data=None
            self.cur_labels=None
            return False
        if self.cur_data is not None:
            
            for x in self.cur_data:
                self.cur_data[x][:-self.hs,:,:]=self.cur_data[x][self.hs:,:,:]
            f_start=self.N-self.hs
        else:            
            f_start=0
            self.cur_data={'p':np.zeros([self.N,data[0]['size'],3]),
                          'r':np.zeros([self.N,data[0]['size'],3]),
                          'q':np.zeros([self.N,data[0]['size'],4])}
            
            
        for f in range(f_start,self.N):        
            for k, d_k in enumerate(data[f]['data']):                
                self.cur_data['p'][f,k,:]=d_k['value']['position']
                self.cur_data['r'][f,k,:]=d_k['value']['rotation']
                self.cur_data['q'][f,k,:]=d_k['value']['quaternion']
        self.cur_data['p'][f_start:,:,:]/=1000.
        d.collect_data('clean_data_time',(time.time()-cT)*1000)                        
        return True    
        
        
    def run(self):
        while True:
            rough_data=self.buffer.acquire()
            
            if self.clean_data(rough_data):        
                cT=time.time()
                self.compute_ft()
                d.collect_data('feature_time',(time.time()-cT)*1000)                
                self.send()
    def send(self):        
        self.sender.set_data(self.out_data)
        self.sender.send()
