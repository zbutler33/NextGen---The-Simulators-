# NextGen---The-Simulators-
Data Assimilation of USGS Data into the NextGen NWM framework 
The Data retriver BMI used nwis python library from https://github.com/USGS-python/dataretrieval
The The ensemble Kalman filter (EnKF) data assimialtion is based on a FilterPy library. FilterPy is a Python library that implements a number of Bayesian filters. """Copyright 2015 Roger R Labbe Jr. FilterPy library.http://github.com/rlabbe/filterpy Documentation at: https://filterpy.readthedocs.org Supporting book at: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python.

Workflow: Use Toy model with fake forcing and catchment data to test DA methodology. Developed 4 BMI's for the Toy Model
- Used 'Toy_Model' Folder
- Used 'USGS' Folder 


1) CFE Peturbed - 'Toy_Model/CFE'
Name: 'bmi_cfe_enkf_peturb.py' which references originial CFE Model, 'cfe_statevars.py' that is able to edit state variables ,'runoff_queue_m_per_timestep' and 'soil_reservoir_storage_deficit_m'
	- BMI references 'config_cfe_peturb_model_CT_calibration.json', which defines catchment characteristics, # of ensembles, and peturbation factor
		- 'cat' file references fake forcing data, 'Toy_Model/Forcing/forcing_data_May17.csv'
	- BMI runs over 7 ensembles with a uniform random distribution of the outflow based on a peturbation factor of 0.75 of the outflow
	- Has state var change option to change state vars based on DA
		- Need to update state vars every DA time step
	- This allows for a covariance matrix and mean to be created from the ensembles, which is needed to run the EnKF DA
	- User can call BMI in a framework, 'State_VarChanges.ipynb' to develop 'look up' table that shows how a changes in state variables will result in change of streamflow
		- Shows linear relationship between outflow and 'runoff_queue_m_per_timestep' -> one to one ratio in 'runoff_queue_m_per_timestep' and no change in 'soil_reservoir_storage_deficit_m'

2) USGS - 'USGS'
Name: 'bmi_usgs.py' which references usgs Model, 'usgs.py'
	- BMI references 'usgs_config.json', which defines site #, output from station, and time range to analyze
	- BMI takes in config information to average hourly outputs of streamflow (instanteous 15min meausrements are averaged every hour)
	- Outputs streamflow

3) DA EnKF - 'Toy_Model/Assimilation'
Name: 'Bmi_da_ENKF_forSBMI.py' which references 'EnKF.py'
	- BMI references 'EnKF_config.json', which defines initial conditions for EnKF 
	- BMI takes in covariance and mean of CFE Peturbed with USGS outflow to produce new outflow
		- The covaraince and mean need to be based on the mean of the state variable change
		- Change state vars in CFE peturbed for DA to read in at every future time step	
	- BMI changes 'runoff_queue_m_per_timestep' and 'soil_reservoir_storage_deficit_m' based on logical statements
		- Determined whether CFE under or over predicts streamflow to then edit state variables
		- Maximum soil storage used to make sure edited state variables are realistic

4) Update State Variable in CFE Analysis - 'Toy_Model/CFE'
Name: 'bmi_cfe_enkf_peturb.py' which referencesCFE Model, 'cfe_statevars.py' that edits state variables ,'runoff_queue_m_per_timestep' and 'soil_reservoir_storage_deficit_m'
	- BMI references 'config_cfe_peturb_model_CT_calibration.json', which defines catchment characteristics
		- 'cat' file references fake forcing data, 'Toy_Model/Forcing/forcing_data_May17.csv'
	- Overall, same BMI as CFE Peturbed, just referencing it differnt in the framework

Framework - 'Toy_Model'
Name: 'Toymodel_V2.ipynb'
	- Loads each CFE, EnKF, and USGS BMI
		- cfe_open (original CFE, nothing changed)
			- Used for model comparison to see if EnKF and CFE Analysis improves streamflow
		- cfe_peturbed (peturbed CFE needed to run EnKF)
		- cfe_analysis (Takes in EnKF data for final assimilated streamflow) 
		- enkf (EnKF methodology using cfe_peturbed and usgs)
		- usgs (take in usgs streamflow data)

 
