import time
import numpy as np
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
import usgs

class BMI_USGS():
    def __init__(self):
        """Create a Bmi CFE model that is ready for initialization."""
        super(BMI_USGS, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        
        # # these need to be initialized here as scale_output() called in update()
        # self.streamflow_cms = 0.0
        # self.streamflow_fms = 0.0
        # self.surface_runoff_m = 0.0

        #----------------------------------------------
        # Required, static attributes of the model
        #----------------------------------------------
        self._att_map = {
            'model_name':         'USGS Data Assimilation',
            'version':            '1.0',
            'author_name':        'Kenneth Ekpetere',
            'grid_type':          'scalar',
            'time_step_size':      1, 
            'time_units':         '1 hour' }
    
        #---------------------------------------------
        # Input variable names (CSDMS standard names)
        #---------------------------------------------
        self._input_var_names = ['sites', 'start', 'end', 'service']
    
        #---------------------------------------------
        # Output variable names (CSDMS standard names)
        #---------------------------------------------
        self._output_var_names = ['Gage_Flow'] #['Date', 'Flow']
        
        #------------------------------------------------------
        # Create a Python dictionary that maps CSDMS Standard
        # Names to the model's internal variable names.
        # This is going to get long, 
        #     since the input variable names could come from any forcing...
        #------------------------------------------------------
        self._var_name_units_map = {
                                'Gage_Flow':['Gage_Flow','cfs'],
                                'sites':['sites','NA'],
                                'start':['start','Day'],
                                'end':['end','Day'],
                                'service':['service','NA'],
                                #--------------   Dynamic inputs --------------------------------
                                'atmosphere_water__time_integral_of_precipitation_mass_flux':['timestep_rainfall_input_m','kg m-2'],
                                'water_potential_evaporation_flux':['potential_et_m_per_s','m s-1'],
                                'DIRECT_RUNOFF':['surface_runoff_depth_m','m'],
                                'GIUH_RUNOFF':['flux_giuh_runoff_m','m'],
                                'NASH_LATERAL_RUNOFF':['flux_nash_lateral_runoff_m','m'],
                                'DEEP_GW_TO_CHANNEL_FLUX':['flux_from_deep_gw_to_chan_m','m']
                          }

    #__________________________________________________________________
    #__________________________________________________________________
    # BMI: Model Control Function
    def initialize(self,cfg_file=None, current_time_step=0):

        #------------------------------------------------------------
        # this is the bmi configuration file
        self.cfg_file = cfg_file

        self.current_time_step=current_time_step

        # ----- Create some lookup tabels from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for long_name in self._var_name_units_map.keys()}
        
        # -------------- Initalize all the variables --------------------------# 
        # -------------- so that they'll be picked up with the get functions --#
        for long_var_name in list(self._var_name_units_map.keys()):
            # ---------- All the variables are single values ------------------#
            # ---------- so just set to zero for now.        ------------------#
            self._values[long_var_name] = 0
            setattr( self, self.get_var_name(long_var_name), 0 )

        ############################################################
        # ________________________________________________________ #
        # GET VALUES FROM CONFIGURATION FILE.                      #
        self.config_from_json()                                    #
        
#         # ________________________________________________
#         # The configuration should let the BMI know what mode to run in (framework vs standalone)
#         # If it is stand alone, then load in the forcing and read the time from the forcig file
#         if self.stand_alone == 1:
#             self.load_forcing_file()
#             self.current_time = pd.Timestamp(self.forcing_data['time'][self.current_time_step])

#         # ________________________________________________
#         # In order to check mass conservation at any time
#         self.reset_volume_tracking()
        
#         # ________________________________________________
#         # initialize simulation constants
#         atm_press_Pa=101325.0
#         unit_weight_water_N_per_m3=9810.0
        
        # ________________________________________________
        # Time control
        # self.time_step_size = 3600
        self.timestep_h = 1
        self.timestep_d = self.timestep_h / 24.0
        self.current_time_step = 0
        self.current_time = self.current_time_step
        
        # Initial value
        self.sites = self.sites
       
        # ________________________________________________
        # Inputs
        self._input_m = 0
        self.outlet_m = 0
        self.potential_et_m_per_s      = 0



        ####################################################################
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        # CREATE AN INSTANCE OF THE USGS FLOW MODEL #
        self.usgs_model = usgs.USGS()
        # ________________________________________________________________ #
        # ________________________________________________________________ #
        ####################################################################
        
    
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def update(self):
        self.usgs_model.run_usgs(self)
        # self.scale_output()

    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # # BMI: Model Control Function
    # def update_until(self, until, verbose=True):
    #     for i in range(self.current_time_step, until):
    #         self.cfe_model.run_cfe(self)
    #         self.scale_output()
    #         if verbose:
    #             print("total discharge: {}".format(self.total_discharge))
    #             print("at time: {}".format(self.current_time))
        
    # BMI: Model Control Function
    def update_until(self, until):
        for i in range(self.current_time_step, until, self.time_step_size):
            self.usgs_model.run_usgs(self)
            self.scale_output()
            self.current_time += self.time_step_size

            if self.current_time >= until:
                break
        
        self.current_time_step = self.current_time
        
     # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________
    # BMI: Model Control Function
    def finalize(self,print_mass_balance=False):

        self.finalize_flow(verbose=print_flow)
        self.reset_usgs_stations()

        """Finalize model."""
        self.usgs_model = None
        self.usgs_state = None
    
    # ________________________________________________
    def reset_usgs_stations(self):
        self.start             = 0
        self.end       = 0
        self.sites       = 0

        return
#     def finalize_flow(self, verbose=True):
        
#         self.flow       = self.start
#         if verbose:            
#             print("\n USGS observed flow for station ",self.sites,"is retrieved!")
#             print()
#         return

    def finalize_flow(self, verbose=True):
        
        self.flow       = self.start
        if verbose:            
            print("\n USGS observed flow for station ",self.sites,"is retrieved!")
            print()
        return
    
    #________________________________________________________
    def config_from_json(self):
        with open(self.cfg_file) as data_file:
            data_loaded = json.load(data_file)

        # ___________________________________________________
        # MANDATORY CONFIGURATIONS
        self.sites                  = data_loaded['sites']
        self.service                = data_loaded['service']
        self.start            = data_loaded['start']
        self.end          = data_loaded['end']

        # _______________________________________________
        # # MANDATORY CONFIGURATIONS
        # self.forcing_file               = data_loaded['forcing_file']
        # self.catchment_area_km2         = data_loaded['catchment_area_km2']
        # self.alpha_fc                   = data_loaded['alpha_fc']
        # self.soil_params                = {}
        # self.soil_params['bb']          = data_loaded['soil_params']['bb']
        # self.soil_params['D']           = data_loaded['soil_params']['D']
        # self.soil_params['depth']       = data_loaded['soil_params']['depth']
        # self.soil_params['mult']        = data_loaded['soil_params']['mult']
        # self.soil_params['satdk']       = data_loaded['soil_params']['satdk']
        # self.soil_params['satpsi']      = data_loaded['soil_params']['satpsi']
        # self.soil_params['slop']        = data_loaded['soil_params']['slop']
        # self.soil_params['smcmax']      = data_loaded['soil_params']['smcmax']
        # self.soil_params['wltsmc']      = data_loaded['soil_params']['wltsmc']
        # self.max_gw_storage             = data_loaded['max_gw_storage']
        # self.Cgw                        = data_loaded['Cgw']
        # self.expon                      = data_loaded['expon']
        # self.gw_storage                 = data_loaded['gw_storage']
        # self.soil_storage               = data_loaded['soil_storage']
        # self.K_lf                       = data_loaded['K_lf']
        # self.K_nash                     = data_loaded['K_nash']
        # self.nash_storage               = np.array(data_loaded['nash_storage'])
        # self.giuh_ordinates             = np.array(data_loaded['giuh_ordinates'])

        # ___________________________________________________
        # OPTIONAL CONFIGURATIONS
        if 'stand_alone' in data_loaded.keys():
            self.stand_alone                    = data_loaded['stand_alone']
        if 'forcing_file' in data_loaded.keys():
            self.reads_own_forcing              = True
            self.forcing_file                   = data_loaded['forcing_file']
        # if 'unit_test' in data_loaded.keys():
        #     self.unit_test                      = data_loaded['unit_test']
        #     self.compare_results_file           = data_loaded['compare_results_file']
         
        return

    

#     #-------------------------------------------------------------------
#     #-------------------------------------------------------------------
#     # BMI: Model Information Functions
#     #-------------------------------------------------------------------
#     #-------------------------------------------------------------------
    
    def get_attribute(self, att_name):
    
        try:
            return self._att_map[ att_name.lower() ]
        except:
            print(' ERROR: Could not find attribute: ' + att_name)

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return self.get_attribute( 'model_name' ) #JG Edit

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def get_value(self, var_name):
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ptr(var_name)

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name):
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        # JG Edit
        return self.get_value_ptr(long_var_name)  #.dtype
    
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # JG Edit
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id  

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
#        return np.dtype(self.get_var_type(name)).itemsize
        return np.array(self.get_value(name)).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        # JG Edit
        # all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    #-------------------------------------------------------------------
    # JG Note: what is this used for?
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return self._start_time #JG Edit

    #-------------------------------------------------------------------
    def get_end_time( self ):

        return self._end_time #JG Edit


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self.current_time

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self.get_attribute( 'time_step_size' ) #JG: Edit

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return self.get_attribute( 'time_units' ) 
       
    #-------------------------------------------------------------------
    def set_value(self, var_name, value):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
              Array of new values.
        """ 
        setattr( self, self.get_var_name(var_name), value )
        self._values[var_name] = value

    #------------------------------------------------------------ 
    def set_value_at_indices(self, name, inds, src):
        """Set model values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        # JG Note: TODO confirm this is correct. Get/set values ~=
#        val = self.get_value_ptr(name)
#        val.flat[inds] = src

        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(name)).flatten().shape[0] == 1:
            self.set_value(name, src)
        else:
            # JMFrame: Need to set the value with the updated array with new index value
            val = self.get_value_ptr(name)
            for i in inds.shape:
                val.flatten()[inds[i]] = src[i]
            self.set_value(name, val)

    #------------------------------------------------------------ 
    def get_var_nbytes(self, long_var_name):
        """Get units of variable.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        # JMFrame NOTE: Had to import sys for this function
        return sys.getsizeof(self.get_value_ptr(long_var_name))

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name, dest, indices):
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        indices : array_like
            Array of indices.
        Returns
        -------
        array_like
            Values at indices.
        """
        #JMFrame: chances are that the index will be zero, so let's include that logic
        if np.array(self.get_value(var_name)).flatten().shape[0] == 1:
            return self.get_value(var_name)
        else:
            val_array = self.get_value(var_name).flatten()
            return np.array([val_array[i] for i in indices])

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------ 
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
 
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0: 
            return 1

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
       
        # JG Edit
        # 0 is the only id we have
        if grid_id == 0:
            return 1

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id=0):

        # JG Edit
        # 0 is the only id we have        
        if grid_id == 0:
            return 'scalar'

    #------------------------------------------------------------ 
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    #------------------------------------------------------------ 
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    #------------------------------------------------------------ 
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 

