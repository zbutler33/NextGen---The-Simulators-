import dataretrieval.nwis as nwis
import pandas as pd
pd.options.mode.chained_assignment = None
class USGS():
    def __init__(self):
        super(USGS, self).__init__()
    def run_usgs(self,u):
        sites=u.sites
        service=u.service
        start=u.start
        end=u.end
        site=nwis.get_record(sites=u.sites,service=u.service,start=u.start,end=u.end)
        site.reset_index(inplace=True) #reset index to grab station date
        site['datetime'] = pd.to_datetime(site['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') #transfer to utc so same time throughout
        site_quarter = site.iloc[:,[0,4]] #locates every row by the columns we want (date and flow)
        site_quarter.columns 
        #print(site_quarter)
        site_copy = site.copy()
        site_copy['datetime'] = pd.to_datetime(site_copy['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') #convert datetime to average every hour 
        #df_copy.reset_index(inplace=True) #reset indexes so can grab "Date" column
        #df_copy.drop("datetime", axis=1, inplace=True) #drop unneccsary date column now 
        site_copy.index = site_copy['datetime'] # index so can pull date time in resample
        site_avg = site_copy.resample('H').mean() # Average every hour based on datetime
        site_avg.reset_index(inplace=True) #reset index again to have datetime 
        site_avgflow = site_avg.iloc[:,[0,2]] #locates every row by the columns we want (date and flow)
        site_avgflow.columns = ['Date', 'Flow']
        
        #check validity of extracted data
        site_avgflow.loc[site_avgflow['Flow'] >= 0, 'validity']=1 # if value positive, consider
        site_avgflow.loc[site_avgflow['Flow'] <0,'validity']=0 # if less than zero, not realistic
        site_avgflow.loc[site_avgflow['Flow'].isnull()==True, 'validity']=0 # if NaN not availible
        
        #Output results to csv file
        Flow = site_avgflow['Flow']
        site_avgflow.to_csv('USGS_'+str(sites)+'_obs_streamflow.csv', index=False)
        #site_avgflow.to_csv('USGS_streamflow_for_site_.csv', index=False)
        # to check if the code runs on the framework
        print(Flow) 
        print("USGS station ID",sites)

        return Flow
        


