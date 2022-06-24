from dataretrieval import nwis, utils, codes
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
    def run_usgs(sites,service,start,end):
        site=nwis.get_record(sites=sites,service=service,start=start,end=end)
        print(site['peak_va'])
        return
    