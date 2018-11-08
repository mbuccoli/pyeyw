import time
import random
import socket
import json
import struct
from threading import Thread, Semaphore


import numpy as np

#class StoppableThread

class Debug:
    def __init__(self, debug=True):
        self.debug=debug
        self.data={}
    def set_elements(self, **kwargs):
        self.__dict__.update(kwargs)
    def msg(self, msg):
        if not self.debug:
            return
        print(str(msg))
    def unset(self):
        self.debug=False
    def set(self):
        self.debug=True
            
    def collect_data(self, key, value):
        if not self.debug:
            return
        if key not in self.data:
            self.data[key]=[]        
        else:
            self.data[key].append(value)
    def get_stats(self, key=None):
        if key is None:
            key=list(self.data.keys())
        elif type(key) is not list:
            key=[key]
        for k in key:
            x=self.data[k]
            print('%s\tMean:%.3f\tStd:%.3f\tMin:%.3f\tMax:%.3f'\
                    %(k,np.mean(x), np.std(x),np.min(x),np.max(x)))
    def get_comments(self):
        fs=np.floor(np.mean(self.data['rec_freq']))
        bfs=np.mean(self.data['send_freq'])
        msg='We are receiving data at around %.2f Hz '%fs
        msg+='and storing them into %d-sample buffers.\n'%self.buffer.N
        msg+='Since hopsize is %d, we should a new buffer at %.2f Hz '\
                %(self.buffer.hopsize,fs/self.buffer.hopsize)
        msg+=' which is about the send frequency %.2f Hz.\n'%bfs
        msg+='Moreover, we create a new buffer every %.2f ms '\
                %float(1000*self.buffer.hopsize/fs)
        msg+='which must be higher than the sum of the time required for the following operations:\n'
        operations={'Clean data':'clean_data_time','Buffering':'buffer_time','Features':'feature_time'}
        
        msg+= ' + \n'.join(['\t%s:\t%.2f '%(op,np.mean(self.data[operations[op]])) for op in operations])
        val_sum=np.sum([np.mean(self.data[operations[op]]) for op in operations])
        msg+='=\n\t\t\t\t%.2f ms\n'%val_sum
        max_t_ms=1000*self.buffer.hopsize/fs
        if max_t_ms>val_sum:
            msg+='Luckily, this is exactly what usually happens'
        else:
            msg+='Unfortunately, this usually does not happen'
        val_sum_max=np.sum([np.max(self.data[operations[op]]) for op in operations])
        if max_t_ms>val_sum:
            msg+=', even in the worst case scenario!\n'
            ws=False
            msg+='With this processing time, we may set the hopsize down to %d samples'\
                    %int(np.ceil(fs*val_sum_max/1000.))
            
        else:
            msg+='\nIn the worst case scenario, this does not happen :('
            ws=True #worst scenario
        ops=[op for _, op in operations.items()]
        if ws and len(ops)==2: #computing the probability of worst scenario
            t_max=np.max([np.max(self.data[op]) for op in ops])
            t=np.linspace(0,t_max,1000)
            f=lambda t, mu, std: 1/(std*np.sqrt(2*np.pi)) *np.exp(-.5*np.power(t-mu,2)/std**2)
            y=[]
            for op in ops:
                mu=np.mean(self.data[op])
                std=np.std(self.data[op])
                if std>0:
                    y.append(f(t,mu,std))
                else:
                    m=np.min(np.abs(t-mu))
                    i=np.argmin(np.abs(t-mu))
                    y_=np.zeros(t.shape)
                    if m<.01:                    
                        y_[i]=1
                    y.append(y_)
            
            T=np.tile(t[:,np.newaxis],(1,t.size))
            T=T+T.T
            idx1,idx2=np.where(T>=max_t_ms)
            p=np.sum(y[0][idx1]*y[1][idx2])
            print(p)
            msg+='\n'
            if p<.15:
                msg+="Luckily, "
            else:
                msg+="Unfortunately, "            
            msg+= "this is going to happen %.1f %% of the times"%p*100
                
                
            
        
        print(msg)
            
        
d=Debug()

class Server(Semaphore, Thread):    
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
    def __init__(self, server, Nbuffer=64, hopsize=32):
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
        self.server=server
        self.lT=time.time()
    def run(self):
        while True:
            in_data=self.server.acquire()
            
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



class DummyFeature(FeatureProcess):
    def __init__(self, **kwargs):
        FeatureProcess.__init__(self, **kwargs)
        #self.f=2*(2*np.pi)
        #self.t0=time.time()
        self.Nfft=int(self.N/2+1)
        self.main_idxs=None
        self.main_labels=None
        self.main_points=None
        self.main_fft3D=None
        self.centroid=None
        self.f=np.linspace(0,1,self.Nfft)        
        
    def find_main_points(self):        
        labels=['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']
        compute=True
        if self.main_idxs is not None:
            compute=not np.all([self.main_labels[i]==self.cur_labels[idx] \
                                for i, idx in enumerate(self.main_idxs)])
            
            
        if compute:
            self.main_idxs=[ [j for j, label in enumerate(self.cur_labels) if label.endswith(l)][0] for l in labels]
            self.main_labels=[self.cur_labels[j] for j in self.main_idxs]
            
        self.main_points=self.cur_data['p'][:,self.main_idxs,:]      
        
    def compute_fft3D(self):        
        norm=np.linalg.norm(np.array(self.main_points),axis=2)        
        self.main_fft3D=np.log10(1e-6+np.abs(np.fft.fft(norm,axis=0)[:self.Nfft,:]))
        
    def compute_norm_centroid(self):        
        x=self.main_fft3D[1:,:]
        self.centroid=np.sum(self.f[1:]*np.sum(x,axis=1))/np.sum(x.flatten())
        
    def compute_ft(self):
        #self.out_data=np.random.random()
        #return
        #idxHips=[k for k, l in enumerate(self.cur_labels)  \
        #                     if l.lower().endswith('hips')][0]
        #self.out_data=.5+np.mean(self.cur_data['r'][:,idxHips,1],axis=0)/(2*np.pi)
        #self.out_data=(np.max(self.cur_data['p'][:,idxHips,1],axis=0)-.5)/1.5
        #self.out_data=.5+.5*np.cos(self.f*(time.time()-self.t0))
        
        self.find_main_points()
        self.compute_fft3D()
        self.compute_norm_centroid()
        self.out_data=(np.tanh(6*(1-2*self.centroid))+1)/2

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
                

if __name__=='__main__':
    server=Server("127.0.0.1",8017)
    sender=Sender("127.0.0.1",8050)
    buffer=Buffer(server,Nbuffer=128,hopsize=8)
    fp=DummyFeature(buffer=buffer, sender=sender)
    
    d.set_elements(buffer=buffer)
    server.start()
    #sender.start()
    buffer.start()
    fp.start()
    
            

    '''
    server.join()
    sender.join()
    buffer.join()
    fp.join()
    '''