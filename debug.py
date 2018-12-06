import numpy as np

class Debug:    
    
    def __init__(self, debug=True):
        Debug.debug=debug
        Debug.data={}
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
        fs=np.floor(np.mean(self.data['rec_freq_8017']))
        bfs=np.mean(self.data['send_freq_8050'])
        msg='We are receiving data at around %.2f Hz '%fs
        msg+='and storing them into %d-sample buffers.\n'%self.buffer.N
        msg+='Since hopsize is %d, we should create a new buffer at %.2f Hz '\
                %(self.buffer.hopsize,fs/self.buffer.hopsize)
        msg+=' which is about the send frequency %.2f Hz.\n'%bfs
        msg+='Moreover, we create a new buffer every %.2f ms '\
                %float(1000*self.buffer.hopsize/fs)
        msg+='which must be higher than the sum of the time required for the following operations:\n'
        operations={'Clean data':'clean_data_time','Buffering':'buffer_time_128','Features':'feature_time'}
        
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