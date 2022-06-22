from dataretrieval import nwis, utils, codes
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
    def run_usgs(self, sites):
        nwis.get_record(sites='03339000', service='peaks',start='1970-01-01',end='1990-01-01')
        sites.streamflow = df['peak_va']
        # ________________________________________________
        sites.current_time_step += 1
        sites.current_time      += sites.time_step_size

        return
    