import time
import numpy as np

from network import Receiver, Sender
from features import FeatureProcess
from debug import d
from buffer import Buffer


#class StoppableThread
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
        self.find_main_points()
        self.compute_fft3D()
        self.compute_norm_centroid()
        self.out_data=(np.tanh(6*(1-2*self.centroid))+1)/2

                

if __name__=='__main__':
    receiver=Receiver("127.0.0.1",8017)
    sender=Sender("127.0.0.1",8050)
    buffer=Buffer(receiver,Nbuffer=128,hopsize=8)
    fp=DummyFeature(buffer=buffer, sender=sender)
    
    d.set_elements(buffer=buffer)
    receiver.start()
    #sender.start()
    buffer.start()
    fp.start()
    