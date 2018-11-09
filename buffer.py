import time
import json

from threading import Thread, Semaphore
from debug import d

def deep_copy(obj):
    if type(obj) in [int, str,float]:
        return obj
    if type(obj)==list:
        copy_obj=[]
        for o in obj:
            copy_obj.append(deep_copy(o))
    elif type(obj)==dict:
        copy_obj={}
        for o in obj:
            copy_obj[o]=deep_copy(obj[o])
    else:
        print(type(obj))
    return copy_obj
            

class Buffer(Thread, Semaphore):
    def __init__(self, receiver, Nbuffer=64, hopsize=32):
        Semaphore.__init__(self, value=0)
        Thread.__init__(self)
        
        self.idx=0
        self.buffer=None
        self.N=Nbuffer
        self.__buffer=[0 for i in range(self.N)]
        if type(hopsize)==float:
            self.hopsize=int(hopsize*self.N)
        else:
            self.hopsize=hopsize
        d.msg(self.hopsize)
        self.receiver=receiver
        self.lT=time.time()
    def run(self):
        while True:
            in_data=self.receiver.acquire()
            
            clean_data=json.loads(in_data.decode('utf-8')[:-1])
            self.__buffer[0:-1]=self.__buffer[1:]
            self.__buffer[-1]=clean_data
            self.idx+=1
            if self.idx>=self.N and self.idx%self.hopsize==0:
                cT=time.time()
                
                #self.buffer=deep_copy(self.__buffer)                
                self.buffer=self.__buffer.copy()                
                #d.msg('%d %.3f ms\n'%(self.idx,time.time()-self.lT)*1000)
                self.idx=self.idx%self.N + self.N
                d.collect_data('buffer_time',(time.time()-cT)*1000)                
                self.release()
            
    def acquire(self,**kwargs):
        Semaphore.acquire(self, **kwargs)
        #
        self.lT=time.time()
        return self.buffer
    