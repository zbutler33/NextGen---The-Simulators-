''''
This code will assimilated CFE model data to CFE Peturbed data in our Sub-Region example model

This framework will build in complication but start simple by 'nudging'

'''


class Assimilation():
    def __init__(self):
        super(Assimilation, self).__init__()
        self.simulated = cfe
        self.observed = peturb
        self.assimilation = []

    def run_assimilation(self, peturb, cfe): #two inputs of petrubed CFE and model CFE

        self.average = np.sum()

        return
        

