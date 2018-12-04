import time
import numpy as np

from network import Receiver, Sender
from features import FeatureProcess, DummyFeature, Fluidity_Heaviness
from debug import d
from buffer import Buffer


#class StoppableThread
                

if __name__=='__main__':
    receiver_data=Receiver("127.0.0.1",8017)
    receiver_ft=Receiver("127.0.0.1",9017)
    
    sender=Sender("127.0.0.1",8050)
    buffer=Buffer(receiver_data,Nbuffer=128,hopsize=8)
    feature_trigger=Buffer(receiver_ft,Nbuffer=1,hopsize=1,type_in="str")
    
    data_driven_mq=Fluidity_Heaviness(buffer=buffer, \
                                      sender=sender, \
                                      trigger=feature_trigger,\
                                      models={'Fluidity':'model_fluidity.pickle'})
        
    d.set_elements(buffer=buffer)
    receiver_data.start()
    receiver_ft.start()
    
    sender.start()
    buffer.start()
    feature_trigger.start()
    data_driven_mq.start()
    