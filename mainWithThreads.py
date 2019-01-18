import time
import numpy as np
import os
from network import Receiver, Sender
from features import FeatureProcess, DummyFeature, Fluidity_Heaviness
from debug import d
from buffer import Buffer



#class StoppableThread
                

if __name__=='__main__':
    
    this_dir=os.path.dirname(os.path.abspath(__file__))
    HOPSIZE=2
    FR=30.
    receiver_data=Receiver("127.0.0.1",8017)
    receiver_ft=Receiver("127.0.0.1",9017)
    
    sender=Sender("127.0.0.1",8050,freq=FR/HOPSIZE)
    buffer=Buffer(receiver_data,Nbuffer=128,hopsize=HOPSIZE)
    feature_trigger=Buffer(receiver_ft,Nbuffer=1,hopsize=1,type_in="str")
    
    data_driven_mq=Fluidity_Heaviness(buffer=buffer, \
                                      sender=sender, \
                                      trigger=feature_trigger,\
                                      models={'Fluidity':os.path.join(this_dir,'model_fluidity.pickle')})
        
    d.set_elements(buffer=buffer)
    receiver_data.start()
    receiver_ft.start()
    
    
    buffer.start()
    feature_trigger.start()
    data_driven_mq.start()
    sender.start()
    x=""
    while x!="stop":
        x=input("\n> ")
        if x=="debug":
            d.get_comments()
    os._exit(0)