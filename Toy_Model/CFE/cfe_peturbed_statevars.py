import time
import numpy as np
import pandas as pd

"""
Summary
--------
This code is built on the original CFE model but with added peturbation and state variable updates for data assimilation.
The updated state variables are "soil_reservoir_storage_deficit_m" and "surface_runoff_depth_m". 
They are updated by multiplying a "time_state_var_change_runoff". 
    - The changing of "soil_reservoir_storage_deficit_m" is done by logical statements based on how much water the soil can take in Bmi_da_ENKF_forSBMI.py
Based on the EnKF updated streamflow, the state variables can be updated each timestep for proper data assimilation technique.
    - See "Bmi_da_ENKF_forSBMI.py" for how each state variable is explicitly updated. 

The added peturbation of CFE is needed to obtain the covariance and mean of 7 ensembles based on a peturbation factor of the outflow of CFE.
The covariance and mean are needed to run the EnKF data assimilation method. 

There is no evapotranspiration in this model. This might cause unrealisitic results, but is ET is not needed to test the cahanges made here. 

Contact
--------
Zach Butler. butlerz@oregonstate.edu
Fitsume Teshome Wolkeba. fwolkeba@crimson.ua.edu  

Inputs
--------
"time_state_var_change_runoff" from EnKF data assimilation.
"self.number_of_ensemble" and "self.peturbation_factor" from config file.

Outputs
--------
"self.surface_runoff_m_ens" 7 outflows from peturbed. Get mean and covariance to feed into EnKF in framework.
"soil_reservoir_storage_deficit_m" and "surface_runoff_depth_m" Updated state variable for final data assimilated streamflow. 

References
----------
https://github.com/zbutler33/NextGen---The-Simulators- 

"""

class CFE():
    def __init__(self):
        super(CFE, self).__init__()
        
    # __________________________________________________________________________________________________________
    # MAIN MODEL FUNCTION
    def run_cfe(self, cfe_state):
        
        # ________________________________________________
        cfe_state.volin += cfe_state.timestep_rainfall_input_m
        
        # ________________________________________________
        cfe_state.potential_et_m_per_timestep = cfe_state.potential_et_m_per_s * cfe_state.time_step_size
        
        # ________________________________________________
        # SUBROUTINE
        # timestep_rainfall_input_m = f(timestep_rainfall_input_m, potential_et_m_per_timestep)
        self.et_from_rainfall(cfe_state)
        
        #_________________________________________________
        # Updating soil_reservoir_storage_deficit_m based on state variable changed from EnKF data assimilation
        cfe_state.soil_reservoir_storage_deficit_m = (cfe_state.soil_params['smcmax'] * \
                                                 cfe_state.soil_params['D'] - \
                                                 cfe_state.soil_reservoir['storage_m'])         
                                                 
        #_________________________________________________
        # soil reservoir storage cannot be negative 
        if cfe_state.soil_reservoir_storage_deficit_m < 0:
            cfe_state.soil_reservoir_storage_deficit_m = 0

        # ________________________________________________
        # SUBROUTINE
        # Calculates the value for surface_runoff_depth_m
        self.Schaake_partitioning_scheme(cfe_state)

        # ________________________________________________
        # SUBROUTINE
        self.et_from_soil(cfe_state)
        #cfe_state.vol_et_from_soil += cfe_state.actual_et_m_per_timestep
        cfe_state.volout += cfe_state.actual_et_m_per_timestep

        # ________________________________________________
        if cfe_state.soil_reservoir_storage_deficit_m < cfe_state.infiltration_depth_m:
            cfe_state.surface_runoff_depth_m += (cfe_state.infiltration_depth_m - cfe_state.soil_reservoir_storage_deficit_m)
            cfe_state.infiltration_depth_m = cfe_state.soil_reservoir_storage_deficit_m
            cfe_state.soil_reservoir['storage_m'] = cfe_state.soil_reservoir['storage_max_m']
        #print(cfe_state.infiltration_depth_m)

        # ________________________________________________
        cfe_state.vol_sch_runoff += cfe_state.surface_runoff_depth_m
        cfe_state.vol_sch_infilt += cfe_state.infiltration_depth_m

        # ________________________________________________
        if cfe_state.current_time_step == 0:
            cfe_state.previous_flux_perc_m = cfe_state.flux_perc_m
            
        # ________________________________________________
        if cfe_state.previous_flux_perc_m > cfe_state.soil_reservoir_storage_deficit_m:
            diff = cfe_state.previous_flux_perc_m - cfe_state.soil_reservoir_storage_deficit
            cfe_state.infiltration_depth_m = cfe_state.soil_reservoir_storage_deficit_m
            cfe_state.vol_sch_runoff += diff
            cfe_state.vol_sch_infilt -= diff
            cfe_state.surface_runoff_depth_m += diff
            
        # ________________________________________________
        cfe_state.vol_to_soil += cfe_state.infiltration_depth_m
        cfe_state.soil_reservoir['storage_m'] += cfe_state.infiltration_depth_m

        # ________________________________________________
        # SUBROUTINE
        # primary_flux, secondary_flux = f(reservoir)
        self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.soil_reservoir)

        # ________________________________________________
        cfe_state.flux_perc_m = cfe_state.primary_flux_m
        cfe_state.flux_lat_m = cfe_state.secondary_flux_m

        # ________________________________________________
        cfe_state.gw_reservoir_storage_deficit_m = cfe_state.gw_reservoir['storage_max_m'] - cfe_state.gw_reservoir['storage_m']
        
        # ________________________________________________
        if cfe_state.flux_perc_m > cfe_state.gw_reservoir_storage_deficit_m:
            diff = cfe_state.flux_perc_m - cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.flux_perc_m = cfe_state.gw_reservoir_storage_deficit_m
            cfe_state.vol_sch_runoff+=diff 
            cfe_state.vol_sch_infilt-=diff 
            
        # ________________________________________________
        cfe_state.vol_to_gw                += cfe_state.flux_perc_m
        cfe_state.vol_soil_to_gw           += cfe_state.flux_perc_m
        cfe_state.gw_reservoir['storage_m']   += cfe_state.flux_perc_m
        cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_perc_m
        cfe_state.soil_reservoir['storage_m'] -= cfe_state.flux_lat_m
        cfe_state.vol_soil_to_lat_flow        += cfe_state.flux_lat_m  #TODO add this to nash cascade as input
        cfe_state.volout                       = cfe_state.volout + cfe_state.flux_lat_m;
    
        # ________________________________________________
        # SUBROUTINE
        # primary_flux, secondary_flux = f(reservoir)
        self.conceptual_reservoir_flux_calc(cfe_state, cfe_state.gw_reservoir) 
            
        # ________________________________________________
        cfe_state.flux_from_deep_gw_to_chan_m = cfe_state.primary_flux_m
        cfe_state.vol_from_gw += cfe_state.flux_from_deep_gw_to_chan_m
        
        # ________________________________________________        
        if not self.is_fabs_less_than_epsilon(cfe_state.secondary_flux_m, 1.0e-09):
            print("problem with nonzero flux point 1\n")
                        
        # ________________________________________________                               
        cfe_state.gw_reservoir['storage_m'] -= cfe_state.flux_from_deep_gw_to_chan_m
        
        # ________________________________________________
        # SUBROUTINE
        # giuh_runoff_m = f(Schaake_output, giuh_ordinates, runoff_queue_m_per_timestep)
        self.convolution_integral(cfe_state)
        
        # ________________________________________________
        cfe_state.vol_out_giuh += cfe_state.flux_giuh_runoff_m
        cfe_state.volout += cfe_state.flux_giuh_runoff_m + cfe_state.flux_from_deep_gw_to_chan_m
        
        # ________________________________________________
        # SUBROUTINE
        self.nash_cascade(cfe_state)

        # ________________________________________________
        cfe_state.vol_in_nash += cfe_state.flux_lat_m
        cfe_state.vol_out_nash += cfe_state.flux_nash_lateral_runoff_m
        
        # ________________________________________________
        cfe_state.flux_Qout_m = cfe_state.flux_giuh_runoff_m + cfe_state.flux_nash_lateral_runoff_m + cfe_state.flux_from_deep_gw_to_chan_m #m/hr
        cfe_state.total_discharge = cfe_state.flux_Qout_m * cfe_state.catchment_area_km2 * 1000000.0 / 3600.0 #m3/s
        #cfe_state.total_discharge = cfe_state.flux_Qout_m * cfe_state.catchment_area_km2 * 1000000.0 #m h-1
        
        # ________________________________________________
        cfe_state.current_time_step += 1
        cfe_state.current_time      += pd.Timedelta(value=cfe_state.time_step_size, unit='s')

        return
    
    
    # __________________________________________________________________________________________________________
    def nash_cascade(self,cfe_state):
        """
            Solve for the flow through the Nash cascade to delay the 
            arrival of the lateral flow into the channel
        """
        Q = np.zeros(cfe_state.num_lateral_flow_nash_reservoirs)
        
        for i in range(cfe_state.num_lateral_flow_nash_reservoirs):
            
            Q[i] = cfe_state.K_nash * cfe_state.nash_storage[i]
            
            cfe_state.nash_storage[i] -= Q[i]
            
            if i == 0:
                
                cfe_state.nash_storage[i] += cfe_state.flux_lat_m
                
            else:
                
                cfe_state.nash_storage[i] += Q[i-1]
        
        cfe_state.flux_nash_lateral_runoff_m = Q[cfe_state.num_lateral_flow_nash_reservoirs - 1]
        
        return
    
                               
    # __________________________________________________________________________________________________________
    def convolution_integral(self,cfe_state):
        """
            This function solves the convolution integral involving N GIUH ordinates.
            
            Inputs:
                Schaake_output_runoff_m
                num_giuh_ordinates
                giuh_ordinates
            Outputs:
                runoff_queue_m_per_timestep
        """

        #_________________________________________________
        # Updating surface_runoff_depth_m based on state variable changed from EnKF data assimilation
        for i in range(cfe_state.num_giuh_ordinates):

            cfe_state.runoff_queue_m_per_timestep[i] += cfe_state.giuh_ordinates[i] * (cfe_state.surface_runoff_depth_m * cfe_state.time_state_var_change_runoff)

        cfe_state.flux_giuh_runoff_m = cfe_state.runoff_queue_m_per_timestep[0]
        
        # __________________________________________________________________
        # shift all the entries in preperation for the next timestep
        for i in range(1, cfe_state.num_giuh_ordinates):  
            
            cfe_state.runoff_queue_m_per_timestep[i-1] = cfe_state.runoff_queue_m_per_timestep[i]

        cfe_state.runoff_queue_m_per_timestep[-1] = 0

        return
    
    # __________________________________________________________________________________________________________
    def et_from_rainfall(self,cfe_state):
        
        """
            if it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
        """
        
        if cfe_state.timestep_rainfall_input_m >0.0:

            if cfe_state.timestep_rainfall_input_m > cfe_state.potential_et_m_per_timestep:
        
                cfe_state.actual_et_m_per_timestep = cfe_state.potential_et_m_per_timestep
                cfe_state.timestep_rainfall_input_m -= cfe_state.actual_et_m_per_timestep

            else: 

                cfe_state.potential_et_m_per_timestep -= cfe_state.timestep_rainfall_input_m
                cfe_state.timestep_rainfall_input_m=0.0
        return
                
                
    # __________________________________________________________________________________________________________
    ########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
    ##########                -or-                 ###############
    ##########    TWO OUTLET NONLINEAR RESERVOIR   ###############                        
    def conceptual_reservoir_flux_calc(self,cfe_state,reservoir):
        """
            This function calculates the flux from a linear, or nonlinear 
            conceptual reservoir with one or two outlets, or from an
            exponential nonlinear conceptual reservoir with only one outlet.
            In the non-exponential instance, each outlet can have its own
            activation storage threshold.  Flow from the second outlet is 
            turned off by setting the discharge coeff. to 0.0.
        """

        if reservoir['is_exponential'] == True: 
            flux_exponential = np.exp(reservoir['exponent_primary'] * \
                                      reservoir['storage_m'] / \
                                      reservoir['storage_max_m']) - 1.0
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * flux_exponential
            cfe_state.secondary_flux_m=0.0
            return
    
        cfe_state.primary_flux_m=0.0
        
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_primary_m']
        
        if storage_above_threshold_m > 0.0:
                               
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_primary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_primary'])
            
            cfe_state.primary_flux_m = reservoir['coeff_primary'] * storage_power

            if cfe_state.primary_flux_m > storage_above_threshold_m:
                cfe_state.primary_flux_m = storage_above_threshold_m
                
        cfe_state.secondary_flux_m = 0.0
            
        storage_above_threshold_m = reservoir['storage_m'] - reservoir['storage_threshold_secondary_m']
        
        if storage_above_threshold_m > 0.0:
            
            storage_diff = reservoir['storage_max_m'] - reservoir['storage_threshold_secondary_m']
            storage_ratio = storage_above_threshold_m / storage_diff
            storage_power = np.power(storage_ratio, reservoir['exponent_secondary'])
            
            cfe_state.secondary_flux_m = reservoir['coeff_secondary'] * storage_power
            
            if cfe_state.secondary_flux_m > (storage_above_threshold_m - cfe_state.primary_flux_m):
                cfe_state.secondary_flux_m = storage_above_threshold_m - cfe_state.primary_flux_m
                
        return
    
    
    # __________________________________________________________________________________________________________
    #  SCHAAKE RUNOFF PARTITIONING SCHEME
    def Schaake_partitioning_scheme(self,cfe_state):
        """
            This subtroutine takes water_input_depth_m and partitions it into surface_runoff_depth_m and
            infiltration_depth_m using the scheme from Schaake et al. 1996. 
            !--------------------------------------------------------------------------------
            modified by FLO April 2020 to eliminate reference to ice processes, 
            and to de-obfuscate and use descriptive and dimensionally consistent variable names.
            
            inputs:
              timestep_d
              Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
              column_total_soil_moisture_deficit_m (soil_reservoir_storage_deficit_m)
              water_input_depth_m (timestep_rainfall_input_m) amount of water input to soil surface this time step [m]
            outputs:
              surface_runoff_depth_m      amount of water partitioned to surface water this time step [m]
              infiltration_depth_m
        """
        
        if 0 < cfe_state.timestep_rainfall_input_m:
            
            if 0 > cfe_state.soil_reservoir_storage_deficit_m:
                
                cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = 0.0
                
            else:
                
                schaake_exp_term = np.exp( - cfe_state.Schaake_adjusted_magic_constant_by_soil_type * cfe_state.timestep_d)
                
                Schaake_parenthetical_term = (1.0 - schaake_exp_term)
                
                Ic = cfe_state.soil_reservoir_storage_deficit_m * Schaake_parenthetical_term
                
                Px = cfe_state.timestep_rainfall_input_m
                
                cfe_state.infiltration_depth_m = (Px * (Ic / (Px + Ic)))
                
                if 0.0 < (cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m):
                    
                    cfe_state.surface_runoff_depth_m = cfe_state.timestep_rainfall_input_m - cfe_state.infiltration_depth_m
                    
                else:
                    
                    cfe_state.surface_runoff_depth_m = 0.0
                    
                    cfe_state.infiltration_depth_m =  cfe_state.timestep_rainfall_input_m - cfe_state.surface_runoff_depth_m
                    
        else:
            
            cfe_state.surface_runoff_depth_m = 0.0
            
            cfe_state.infiltration_depth_m = 0.0
            
        return

    # __________________________________________________________________________________________________________
    def et_from_soil(self,cfe_state):
        """
            take AET from soil moisture storage, 
            using Budyko type curve to limit PET if wilting<soilmoist<field_capacity
        """
        
        '''
        NO ET USED FOR THIS MODEL. TESTING DATA ASSIMILATION

        '''                                          
            
    # __________________________________________________________________________________________________________
    def is_fabs_less_than_epsilon(self,a,epsilon):
        
        if np.abs(a) < epsilon:
            
            return True
        
        else:
            
            return False 
    