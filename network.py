import time
import random
import socket
import struct
from threading import Thread, Semaphore

from debug import d

class Receiver(Semaphore, Thread):    
    def __init__(self, ip, port):
        Semaphore.__init__(self, value=0)        
        self.in_data=None    
        Thread.__init__(self)        
        self.ip=ip
        self.port=port
        self.socket=socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
        self.socket.bind((self.ip, self.port))
        self.lT=time.time()
    
    def run(self):
        while True:
            self.in_data, _= self.socket.recvfrom(48*1024)                        
            d.collect_data('rec_freq_%d'%self.port,1/(time.time()-self.lT+1e-6))  
            d.collect_data('rec_time_ms_%d'%self.port,1000*(time.time()-self.lT))         
            self.lT=time.time()
            #print('received data')
            self.release()
    def acquire(self,**kwargs):
        Semaphore.acquire(self, **kwargs)
        #print('sending data')
        return self.in_data
        

class Sender(Thread,Semaphore):    
    def __init__(self, ip, port,freq=100):                
        
        Thread.__init__(self)        
        Semaphore.__init__(self, value=1)                
        self.port=port
        self.address=(ip, port)
        self.lT=time.time()
        self.time_wait=1./freq
        
        self.socket = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
        

        
        self.out_data=None
        self.status="inactive"
    def set_data(self, data):
        pass
        #print("SD: Waiting for the Semaphore")
        self.acquire()
        self.status="set_data"
    
        if data is None:
            self.out_data=data
        else:
            self.out_data=struct.pack("d",data)
        self.release()
        #print("SD: Released the Semaphore")
    def send(self):
        pass
        #print("SEND: Waiting for the Semaphore")
        self.acquire()
        
        if self.out_data is not None:
            self.status="sending"
            self.socket.sendto(self.out_data, self.address)    
            d.collect_data('send_freq_%d'%self.port,1/(1e-6+time.time()-self.lT))                
            self.lT=time.time()        
        
        self.release()
        time.sleep(self.time_wait)
        #print("SEND: Released the Semaphore")
        
    def run(self):        
            while True:                
                self.send()