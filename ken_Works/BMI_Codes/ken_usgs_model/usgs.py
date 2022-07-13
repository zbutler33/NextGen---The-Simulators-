import dataretrieval.nwis as nwis
from dataretrieval import nwis, utils, codes
import pandas as pd
import matplotlib.pyplot as plt


# class USGS():
#     def __init__(self):
#         super(USGS, self).__init__()

#     # RUNNING USGS SCRIPT
#     def run_usgs(self, k):
#         # site, start='2019-02-01', end='2019-04-01', service='iv'
#         site=k.sites; service=k.service; start=k.service; end=k.service
                
#         # get peak flow # get instantaneous values (iv)
#         station_record = nwis.get_record(site=k.sites, service=k.service, start=k.start, end=k.end)
#         print(station_record['peak_va'])
#         # station_record = station_record['peak_va'] #.rename(['Flow'])
#         # # station_record = station_record.resample('H').mean()

        
#         return 

    
    
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
        
    # __________________________________________________________________________________________________________
    # RUNNING USGS SCRIPT
    def run_usgs(self, k):
        
        # get instantaneous values (iv)
        sites=k.sites; service=k.service; start=k.service; end=k.service
                
        # get peak flow # get instantaneous values (iv)
        station_record = nwis.get_record(sites=k.sites, service=k.service, start=k.start, end=k.end)
        
        station_record.reset_index(inplace=True)
        station_record['datetime'] = pd.to_datetime(station_record['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') 
        station_record_quarter = station_record.iloc[:,[0,4]]
        station_record_quarter.columns = ['Date', 'Flow']
        
        ## Make copy od dataframe to average flow every hour 
        station_record_copy = station_record.copy()
        station_record_copy['datetime'] = pd.to_datetime(station_record_copy['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S')
        station_record_copy.index = station_record_copy['datetime']
        station_record_avg = station_record_copy.resample('H').mean()
        station_record_avg.reset_index(inplace=True)
        
        station_record_avgflow = station_record_avg.iloc[:,[0, 2]] #station_record_avg.iloc[:,[0,2]]
        station_record_avgflow.columns = ['Date', 'Flow']
        print(station_record_avgflow)
        
        return    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# usgs_instance=USGS()
# station_record=usgs_instance.run_usgs(sites='01054500', service='peak_va', start='2018-02-01', end='2018-06-01')
    
# print(station_record)




# class USGS():
#     def __init__(self):
#         super(USGS, self).__init__()
#     def run_usgs(self,u):
#         sites=u.sites
#         service=u.service
#         start=u.service
#         end=u.service
#         site=nwis.get_record(sites=u.sites,service=u.service,start=u.start,end=u.end)
#         print(site['peak_va'])
#         return
   


# class USGS():
#     def __init__(self):
#         super(USGS, self).__init__()
        
#     # __________________________________________________________________________________________________________
#     # RUNNING USGS SCRIPT
#     def run_usgs(self, site, start_date='2019-02-01', end_date='2019-04-01', type_data='iv'):
        
#         # get instantaneous values (iv)
#         df = nwis.get_record(sites=site, service=type_data, start=start_date, end=end_date)
        
#         df.reset_index(inplace=True)
#         df['datetime'] = pd.to_datetime(df['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') 
#         df_quarter = df.iloc[:,[0,4]]
#         df_quarter.columns = ['Date', 'Flow']
        
#         ## Make copy od dataframe to average flow every hour 
#         df_copy = df.copy()
#         df_copy['datetime'] = pd.to_datetime(df_copy['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S')
#         df_copy.index = df_copy['datetime']
#         df_avg = df_copy.resample('H').mean()
#         df_avg.reset_index(inplace=True)
        
#         df_avgflow = df_avg.iloc[:,[0,2]]
#         df_avgflow.columns = ['Date', 'Flow']
        
#         return df_avgflow 

# usgs_instance=USGS()
# df=usgs_instance.run_usgs(site='01054500', start_date='2018-02-01', end_date='2018-06-01')
    
# print(df)