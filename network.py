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
            d.collect_data('rec_freq',1/(time.time()-self.lT))  
            d.collect_data('rec_time_ms',1000*(time.time()-self.lT))         
            self.lT=time.time()
            #print('received data')
            self.release()
    def acquire(self,**kwargs):
        Semaphore.acquire(self, **kwargs)
        #print('sending data')
        return self.in_data
        

class Sender(Thread,Semaphore):    
    def __init__(self, ip, port):                
        self.THROUGH_EYW=False
        Thread.__init__(self)        
        Semaphore.__init__(self, value=1)                
        self.address=(ip, port)
        self.lT=time.time()
        self.socket = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
        self.fromEyw=None
        if self.THROUGH_EYW:        
            self.fromEyw=socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
            self.fromEyw.bind(("127.0.0.1", 9017))
        self.out_data=None
        self.status="inactive"
    def set_data(self, data):
        self.acquire()
        self.status="set_data"
        if self.THROUGH_EYW:        
            self.out_data,_=self.fromEyw.recvfrom(8)                
        else:
            self.out_data=struct.pack("d",data)
        self.release()
    def send(self):
        self.acquire()
        
        if self.out_data is None:
            self.release()
            return        
        self.status="sending"
        self.socket.sendto(self.out_data, self.address)    
        #print("sending %.2f LT: %.3f ms"%(struct.unpack('d',self.out_data)[0],(time.time()-self.lT)*1000))
        d.collect_data('send_freq',1/(time.time()-self.lT))                
        self.lT=time.time()        
        self.release()
        
    def run(self):        
            while True:                
                self.send()