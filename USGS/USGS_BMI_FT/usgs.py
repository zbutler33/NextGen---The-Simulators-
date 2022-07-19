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
        site0=nwis.get_iv(sites=u.sites,parameterCd="00060",start=u.start,end=u.end)
        site=pd.DataFrame(site0[0])
        
        # print(site[0])
        # site = site[0]["00060"]
        site.reset_index(inplace=True) #reset index to grab station date
        site['datetime'] = pd.to_datetime(site['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') #transfer to utc so same time throughout
        # print(site)
        # site_quarter = site.iloc[:,[0,4]] #locates every row by the columns we want (date and flow)
        # site_quarter.columns 
        #print(site_quarter)
        site_copy = site.copy()
        site_copy['datetime'] = pd.to_datetime(site_copy['datetime'], utc=True, format = '%Y-%m-%d %H:%M:%S') #convert datetime to average every hour 
        #df_copy.reset_index(inplace=True) #reset indexes so can grab "Date" column
        #df_copy.drop("datetime", axis=1, inplace=True) #drop unneccsary date column now 
        site_copy.index = site_copy['datetime'] # index so can pull date time in resample
        site_avg = site_copy.resample('H').mean() # Average every hour based on datetime
        site_avg.reset_index(inplace=True) #reset index again to have datetime 
        # site_avgflow = site_avg.iloc[:,[0,2]] #locates every row by the columns we want (date and flow)
         
        site_avg.columns = ['Date', 'Flow']

        #check validity of extracted data
        site_avg.loc[site_avg['Flow'] >= 0, 'validity']=1 # if value positive, consider
        site_avg.loc[site_avg['Flow'] <0,'validity']=0 # if less than zero, not realistic
        site_avg.loc[site_avg['Flow'].isnull()==True, 'validity']=0 # if NaN not availible
        #Output results to csv file
        site_avg.to_csv('USGS_'+str(sites)+'_obs_streamflow.csv', index=False)
        #site_avgflow.to_csv('USGS_streamflow_for_site_.csv', index=False)
        # to check if the code runs on the framework
        u.flow=site_avg['Flow']
        u.validity=site_avg['validity']
        print(site_avg) 
        print("USGS station ID",sites)

        return 
        


