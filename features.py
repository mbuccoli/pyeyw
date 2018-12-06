import time
import numpy as np
from threading import Thread
from debug import d
import pickle

    
class FeatureProcess(Thread):    
    def __init__(self, buffer,sender,**kwargs):
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
        self.cur_data['p'][f_start:,:,:]/=10#00.
        d.collect_data('clean_data_time',(time.time()-cT)*1000)                        
        return True    
        
        
    def run(self):
        while True:
            #print("FT: acquiring data...")
            rough_data=self.buffer.acquire()
            
            if self.clean_data(rough_data):        
                cT=time.time()
                #print("FT: computing ft...")
                self.compute_ft()
                d.collect_data('feature_time',(time.time()-cT)*1000)                
                #print("FT: sending...")
                self.send()
    def send(self):        
        #print("FT: setting data...")
        self.sender.set_data(self.out_data)
        #print("FT: sending data...")
        #self.sender.send()



class DummyFeature(FeatureProcess):
    def __init__(self, **kwargs):
        FeatureProcess.__init__(self, **kwargs)        
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



class Fluidity_Heaviness(FeatureProcess):
    def __init__(self, **kwargs):
        FeatureProcess.__init__(self, **kwargs)        
        self.Nfft=int(self.N/2+1)
        self.main_idxs=None
        self.main_labels=None
        self.main_points=None
        self.main_fft3D=None
        self.centroid=None
        self.f=np.linspace(0,1,self.Nfft)     
        
        self.trigger=kwargs["trigger"]
        self.models={'Fluidity':kwargs["models"]["Fluidity"]}
        self.features={'Fluidity':.5}
        for ft in self.models:
            model=self.models[ft]
            if type(model) ==str:
                with open(model,'rb') as fp:
                    self.models[ft]=pickle.load(fp)
        print("Loading models...",self.models)
        
        
    
            
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
        
    def predict_fluidity(self):
        #print(self.main_fft3D.shape)
        
        y_p=self.models["Fluidity"].predict(self.main_fft3D.flatten().reshape(1,-1))
        #y_p=self.models["Fluidity"].predict_proba(self.main_fft3D)
        #print('Y_p=',y_p)
        #y_p=np.sum(y_p*np.array([[0,.5,1]]),axis=1).flatten()        
        #print('Y_p sum mul=',y_p)        
        self.features["Fluidity"]=self.features["Fluidity"]*.9+y_p*.05 #/2*0.1
        #print('Self FT=',self.features["Fluidity"])
        
    
    def set_fluidity(self):
        self.out_data=self.features["Fluidity"]
    def set_none(self):
        self.out_data=None
        self.features["Fluidity"]=0.5
        
    def compute_ft(self):        
        #print("FH: finding main points...")
        self.find_main_points()
        mq=self.trigger.buffer
        #print(mq)
        if mq=="Fluidity":
            #print("FH: computing fft...")
            self.compute_fft3D()
            #print("FH: predicting fluidity...")
            self.predict_fluidity()
            #print("FH: set fluidity...")
            self.set_fluidity()
        else:
            self.set_none()
        #self.out_data=(np.tanh(6*(1-2*self.centroid))+1)/2
