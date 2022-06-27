from dataretrieval import nwis, utils, codes
'''
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
    def run_usgs(sites,service,start,end):
        site=nwis.get_record(sites=sites,service=service,start=start,end=end)
        print(site['peak_va'])
        return
'''
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
    def run_usgs(self,u):
        sites=u.sites
        service=u.service
        start=u.service
        end=u.service
        site=nwis.get_record(sites=u.sites,service=u.service,start=u.start,end=u.end)
        print(site['peak_va'])
        return
        

