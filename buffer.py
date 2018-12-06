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
    
    def __init__(self, receiver, Nbuffer=64, hopsize=32, type_in='json'):
        
        Semaphore.__init__(self, value=0)
        Thread.__init__(self)
        
        self.idx=0
        self.buffer=None
        self.N=Nbuffer
        self.__clean_funcs={'json':self.__clean_json,\
                            'str':self.__clean_str}
        self.__clean=self.__clean_funcs[type_in]
        
        if self.N>1:
            self.__buffer=[0 for i in range(self.N)]
            self.__update_buffer=self.__update_buffer_N
        else:
            self.__buffer=0
            self.__update_buffer=self.__update_buffer_1
        if type(hopsize)==float:
            self.hopsize=int(hopsize*self.N)
        else:            
            self.hopsize=hopsize
        #d.msg(self.hopsize)
        self.receiver=receiver
        self.lT=time.time()
    def __update_buffer_N(self,data):
        
        self.__buffer[0:-1]=self.__buffer[1:]
        self.__buffer[-1]=data            
        self.idx+=1
        if self.idx>=self.N and self.idx%self.hopsize==0:
            cT=time.time()
            
            #self.buffer=deep_copy(self.__buffer)                
            self.buffer=self.__buffer.copy()                
            #d.msg('%d %.3f ms\n'%(self.idx,time.time()-self.lT)*1000)
            self.idx=self.idx%self.N + self.N
            d.collect_data('buffer_time_%d'%self.N,(time.time()-cT)*1000)                
            self.release()
        
    def __update_buffer_1(self,data):
        self.__buffer=self.buffer=data
        self.release()
    def __clean_json(self, in_data):
        return json.loads(in_data.decode('utf-8')[:-1])
    def __clean_str(self, in_data):
        return in_data.decode('utf-8')[:-1]
    def run(self):
        while True:
            in_data=self.receiver.acquire()
            
            clean_data=self.__clean(in_data)
            self.__update_buffer(clean_data)
            
            
    def acquire(self,**kwargs):
        Semaphore.acquire(self, **kwargs)
        #
        self.lT=time.time()
        return self.buffer
    
