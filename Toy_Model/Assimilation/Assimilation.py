''''
This code will assimilated CFE model data to CFE Peturbed data in our Sub-Region example model

This framework will build in complication but start simple by 'nudging'

Will build in complexity of data assimilation techniques

'''


class Assimilation():

    #Put two inputs of petrubed CFE (USGS obs) and model CFE into assimilation definition
    def __init__(self, obs, cfe):   
        super(Assimilation, self).__init__()
        
        self.simulated = cfe
        self.observed = obs
        self.assimilation = []

    def run_assimilation(self): 

        assimilation = (self.simulated + self.observed)/2
        self.assimilation = assimilation

        return self.assimilation
    
        

