# NextGen---The-Simulators-
Data Assimilation of USGS Data into the NextGen NWM framework 
The Data retriver BMI used nwis python library from https://github.com/USGS-python/dataretrieval
The The ensemble Kalman filter (EnKF) data assimialtion is based on a FilterPy library. FilterPy is a Python library that implements a number of Bayesian filters. """Copyright 2015 Roger R Labbe Jr. FilterPy library.http://github.com/rlabbe/filterpy Documentation at: https://filterpy.readthedocs.org Supporting book at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.

Workflow: Use Toy model with fake forcing and catchment data to test DA methodology. Developed 4 BMI's for the Toy Model
- Used 'Toy_Model' Folder
- Used 'USGS' Folder 


1) CFE Peturbed - 'Toy_Model/CFE'
Name: 'bmi_cfe_peturb.py' which references originial CFE Model, 'cfe_statevars.py' that is able to edit state variables ,'runoff_queue_m_per_timestep' and 'soil_reservoir_storage_deficit_m'
	- BMI references 'cat_58_config_cfe_peturb_obs.json', which defines catchment characteristics, # of ensembles, and peturbation factor
		- 'cat' file references fake forcing data, 'Toy_Model/Forcing/cat58_01Dec2015.csv'
	- BMI runs over 7 ensembles with a uniform random distribution of the outflow based on a peturbation factor of 0.75 of the outflow
	- Has state var change option to change state vars based on DA
		- Need to update state vars every DA time step
	- This allows for a covariance matrix and mean to be created from the ensembles, which is needed to run the EnKF DA
	- User can call BMI in a framework, 'State_VarChanges.ipynb' to develop 'look up' table that shows how a changes in state variables will result in change of streamflow
		- Might want to develop this 'look up' table in the 'cfe_statevars.py' code
		- Shows linear relationship between outflow and 'runoff_queue_m_per_timestep' -> one to one ratio in 'runoff_queue_m_per_timestep' and no change in 'soil_reservoir_storage_deficit_m'

2) USGS - 'USGS'
Name: 'bmi_usgs.py' which references usgs Model, 'usgs.py'
	- BMI references 'usgs_config.json', which defines site #, output from station, and time range to analyze
	- BMI takes in config information to average hourly outputs of streamflow (instanteous 15min meausrements are averaged every hour)
	- Outputs streamflow

3) Update State Variable - 'Toy_Model/CFE'
Name: 'bmi_cfe_statevars.py' which referencesCFE Model, 'cfe_statevars.py' that edits state variables ,'runoff_queue_m_per_timestep' and 'soil_reservoir_storage_deficit_m'
	- BMI references 'cat_58_config_cfe.json', which defines catchment characteristics
		- 'cat' file references fake forcing data, 'Toy_Model/Forcing/cat58_01Dec2015.csv'
	- Need to call BMI in a framework to develop 'look up' table that shows how a changes in state variables will result in change of streamflow
		- Might want to develop this 'look up' table in the 'cfe_statevars.py' code

4) DA EnKF - 'Toy_Model/Assimilation'
Name: 'Bmi_da_ENKF_forSBMI.py' which references 'EnKF.py'
	- BMI references 'EnKF_config.json', which defines ....
	- BMI takes in covariance and mean of CFE Peturbed with USGS outflow to produce new outflow
		- The covaraince and mean need to be based on the mean of the state variable change
		- Change state vars in CFE peturbed for DA to read in at every future time step	
		- Could give it linear equation or look up table
	- BMI will reference 'factor' that will take in percent change of state variable



