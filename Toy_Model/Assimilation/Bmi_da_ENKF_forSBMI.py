import time
import numpy as np
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
import EnKF
# rename as enkf_cfe_bmi
class EnKF_wrap():
    def __init__(self):
        # super(EnKF_wrap, self).__init__()
        # from code
        #end###############
        """Create a Bmi EnKF data retrieval ready for initialization."""
        super(EnKF_wrap, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo("d").max
        
        #----------------------------------------------
        # Required, static attributes of the model
        #----------------------------------------------
        self._att_map = {
            'model_name':         'ENKF BMI',
            'version':            '1.0',
            'author_name':        '',
            'grid_type':          'scalar',
            'time_units':         '1 hr' }
    
        #---------------------------------------------
        # Input variable names (CSDMS standard names)
        #---------------------------------------------
        self._input_var_names = [
            'x', 'P','z','F', 'dt','N','basin_area_km2','max_state_var_change_soilResDef',
            'soil_storage_avail_m','soil_reservoir_storage_deficit_m','surface_runoff_depth_m']
    
        #---------------------------------------------
        # Output variable names (CSDMS standard names)
        #---------------------------------------------
        self._output_var_names = ['factor','x_prior','x_post','enkf','factor_runoff','factor_soil_res_def','surface_runoff_ratio']    
        
        #------------------------------------------------------
        # Create a Python dictionary that maps CSDMS Standard Names to the model's internal variable names.

        #------------------------------------------------------
        self._var_name_units_map = {"soil_reservoir_storage_deficit_m":['soil_reservoir_storage_deficit_m','m'],
                                    "surface_runoff_depth_m": ['surface_runoff_depth_m', 'm'],
                                    "soil_storage_avail_m":['availible_soil_storage_m','m'],
                                    "basin_area_km2":['basin_area_km2','km2'],
                                    "surface_runoff_ratio":['surface_runoff_ratio','%'],                                   
                                    'max_state_var_change_soilResDef':['depth_of_water','m'],
                                    'F':['percent','%'],'enkf':['enkf_flow','cfs'],
                                    'factor_runoff':['percent','%'],'factor_soil_res_def':['percent','%'],
                                'x_prior':['Qbefore_enkf_update','cfs'],'x_post':['Qafter_enkf_update','cfs'],
                                'x':['mean','cfs'],'z':['obs','cfs'],'factor':['percent','%'],
                                'P':['covariance','NA'],'dt':['time_step','hour'],
                                    'N':['Number_of_ensembles','Day'],
                                    'hx':['observation_function','NA'], 
                                    'fx':['state_transition_funciton','NA']
                          }

    #__________________________________________________________________
    #__________________________________________________________________
    # BMI: Model Control Function

    def initialize(self, cfg_file=None, current_time_step=0):
    #def initialize(self, x, P):
        
        #def initialize(self, cfg_file=None, current_time_step=0):
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
        self.config_from_json()   
        # print(self.N)
        
        # ________________________________________________
        # Time control if incase update until may be used
        self.timestep_h = 1
        self.timestep_d = self.timestep_h / 24.0
        self.current_time_step = 0
        self.current_time = self.current_time_step
        
            # ________________________________________________
        # Initial values for initialising the EnKF BMI
        
        
        
        ########
        # x_post=self.x_post
        self.x=self.x        
        self.P=self.P
        self.z=self.z
        self.F=self.F
        self.dim_z=len(self.z)
        self.N=int(self.N)
        self.dim_x=len(self.x)
        dt=self.dt
        self.N=int(self.N)
        F=self.F
        max_state_var_change_soilResDef=self.max_soil_res_def
        self.basin_area_km2=self.basin_area_km2 
        print("initialized",self.basin_area_km2)
        # print("% chnage of state var",self.L[1][1])
        # x=self.x
        # dt=self.dt

    # CREATE AN INSTANCE OF THE EnKF MODEL #
        self.enkf_model = EnKF.enkf()
        
        # ________________________________________________


    def update(self):
        self.x= self._values['x'] 
        self.P=np.eye(1)*self._values['P']
        self.dt=self._values['dt'] 
        self.F=self._values['F']
        self.z=self._values['z']
        self.surface_runoff = self._values['surface_runoff_depth_m']
        self.soil_storage_deficit = self._values['soil_reservoir_storage_deficit_m']
        self.soil_storage_avail  = self._values['soil_storage_avail_m']
        
        self.basin_area_km2=self.basin_area_km2
        
        self.max_soil_res_def=self._values['max_state_var_change_soilResDef']
        
        self.enkf_model.run_enkf(self)

        self._values['x_post'] = self.x_post
        self._values['x_prior'] = self.x
        self._values['enkf'] = self.res
        self._values['factor'] = self.factor 
        
        # soil_storage_avail_m
        # print("check flot factor",self.factor)
        #################### lookup equation ####################
        enkf_model_diff = (self.x - self.res)
       
        basin_area_m2 =self.basin_area_km2*1000*1000# define config file. move from km to m2

        total_volume_change_ft3 = (enkf_model_diff)*3600
        
        total_volume_change_m3 =(3.28**3)*total_volume_change_ft3 #changing ft3 to m3
        total_volume_change_m =total_volume_change_m3/basin_area_m2

        print(total_volume_change_m)

        ## CFE Underestimation
        if total_volume_change_m > 0:
        # set value of soil_storage_avail_m in the framework from CFE
        # set value of soil moositure deficit
    
            soil_moisture_def_diff_m = total_volume_change_m - self.soil_storage_deficit #leftover soil moisture

            #leftover_depth_change_m = soil_moisture_def_diff_m

            #if soil_moisture_def_diff_m <= 0: #cannot be positive because negative above
            #soil_moisture_def_diff_m = total_volume_change_m + self.soil_storage_deficit
            #self.soil_storage_deficit = total_volume_change_m
            self.soil_storage_avail += total_volume_change_m
            #total_volume_change_m = 0
            leftover_depth_change_m = soil_moisture_def_diff_m
        
            #if soil_moisture_def_diff_m > 0:
            #    leftover_depth_change_m = soil_moisture_def_diff_m
            #    self.soil_storage_deficit  = self.soil_storage_avail # set by CFE

            # Need to check  
            #elif soil_moisture_def_diff_m<0:
                #leftover_depth_change_m=0
                #self.soil_storage_deficit = total_volume_change_m
                #self.soil_storage_avail += total_volume_change_m
                #total_volume_change_m = 0

            # Get ratio of change between runoff queue depth and leftover volume change
            if self.surface_runoff == 0:
                self.surface_runoff_ratio = 1
                self._values['surface_runoff_ratio']=1 #Keep CFE as it is, ratio is 1. 
            else:
                self.surface_runoff_ratio = ((self.surface_runoff + leftover_depth_change_m)/self.surface_runoff)
                self._values['surface_runoff_ratio']=self.surface_runoff_ratio
                self.surface_runoff = self.surface_runoff_ratio*self.surface_runoff
                
            print("Underestimate runoff ratio",self.surface_runoff_ratio)
            

        ## CFE Overestimation
        if total_volume_change_m < 0:

            soil_moisture_def_diff_m = total_volume_change_m - self.soil_storage_deficit #leftover soil moisture
            #if soil_moisture_def_diff_m <= 0:
            #    self.soil_storage_deficit = total_volume_change_m
            #    self.soil_storage_avail += total_volume_change_m
            #    total_volume_change_m = 0
            
            #if soil_moisture_def_diff_m > 0:
            leftover_depth_change_m= soil_moisture_def_diff_m
            self.soil_storage_deficit  = self.soil_storage_avail # set by CFE

            # Need to check  
        # elif soil_moisture_def_diff_m<0:
        #     leftover_depth_change_m=0
        #     self.soil_storage_deficit = total_volume_change_m
        #     self.soil_storage_avail += total_volume_change_m
        #     total_volume_change_m = 0

                # Get ratio of change between runoff queue depth and leftover volume change
            if self.surface_runoff == 0:
                self.surface_runoff_ratio = 1
                self._values['surface_runoff_ratio']=1
            else:
                self.surface_runoff_ratio = ((self.surface_runoff + leftover_depth_change_m)/(self.surface_runoff))
                self._values['surface_runoff_ratio']=self.surface_runoff_ratio
                self.surface_runoff = self.surface_runoff_ratio*self.surface_runoff

            print("Overestimate runoff ratio",self.surface_runoff_ratio)
            
        # original#######################
        # if soil_moisture_def_diff_m >= 0:
        #     self.soil_storage_deficit  = self.soil_storage
        #     self.soil_storage = 0
        #     total_volume_change_m = soil_moisture_def_diff_m
        # elif soil_moisture_def_diff_m < 0:
        #     self.soil_storage_deficit = total_volume_change_m
        #     self.soil_storage += total_volume_change_m
        #     total_volume_change_m = 0
        ###################

            
        

        #self._values['factor_soil_res_def'] = soil_moisture_def_diff_m

        # compute the % change of runoff queue
        #runoff_queue_change_depth=total_volume_change_m #either 0 or soil storage available
        #runoff_queue_volume= runoff_queue_change_depth*basin_area_m2
        #factor_runoff_queue=(runoff_queue_volume/3600)/(enkf_model_diff) # change to m3/sec
        # if enkf prediction is 0, then the factor runoff needs to be 0
        #if self.res == 0:
        #    factor_runoff_queue = 0

        #state_var_change_runoff = factor_runoff_queue 
        #state_var_change_runoff =0.01*(factor_runoff_queue/100) #runoff lookup equation
       
        #print("soil_moisture_def_diff_m",soil_moisture_def_diff_m)
        #self._values['factor_runoff'] = state_var_change_runoff
       

        # change_soil_moisture_def_m
        # percent_change = change_soil_moisture_def_m/total_volume_change_m

        # relate to runoff queue usig the equa.

            # Basin Area in config file Area normalised mass in meters
            # runoff_queue in meters
            # # ########################################
            # state_var_change_runoff=0.01*(factor_runoff_queue)-0 # the remaining factor has to be applied here
            # self._values['factor_runoff'] = state_var_change_runoff
            # self._values['factor_soil_res_def'] = state_var_change_soilResDef
        
        # state_var_change_soilResDef= 49.89247652*(1-self.factor)-49.8818328811 # but this has a physical limit;get.value(max_soil_res_def)
        # print(state_var_change_soilResDef,"check_this")
        # if state_var_change_soilResDef >  self.max_soil_res_def:
        #     factor_soil_res_def=(self.max_soil_res_def/49.89247652)+49.8818328811
        #     factor_runoff_queue =self.factor-factor_soil_res_def
            # total mass change analysis vs mean puturbed; volume
            ##############################################
            # volume = cfs*3600, meter,
            # factor=enkf/CFE
            
        
        
        #######################
        
        # to activate once update is functional
#         L= pd.DataFrame(self.L, columns=["state_Var","streamflow"])

#         change = L.loc[L["streamflow"] == self.factor] # finding similar str. flow change [state_Var %,streamflow%]

#         multiplier= int(change["state_Var"])
#         print("mult",multiplier)
#         print("change",change)
    
        

            #BMI: Model Control Function
####################################################################
    # BMI: Model Control Function if update until may be used (not functional)
    def update_until(self, until):
        for i in range(self.current_time_step, until, self.time_step_size):
            self.enkf_model.run_enkf(self)
            self.scale_output()
            self.current_time += self.time_step_size

            if self.current_time >= until:
                break
        
        self.current_time_step = self.current_time
        
    # __________________________________________________________________________________________________________
    # __________________________________________________________________________________________________________        


    # BMI: Model Control Function
    def finalize(self,print_output=False):

        self.finalize_enkf(verbose=print_output)
        self.reset_enkf_inputs()

        """Finalize model."""
        self.enkf_model = None
        self.enkf_state = None
    
    # ________________________________________________
    def reset_enkf_inputs(self):
        self.x             = 0
        self.P       = 0
        self.z       = 0
        self.F       = 0
        self.dt       = 0
        self.N       = 0

        return
    def finalize_enkf(self, verbose=True):
        
        self.x       = self.P
        if verbose:            
            print("\n Data assimilation completed!")
            print()
        return self.x
#     def scale_output(self): #set output with doc string 

#         #
#         self._values['L'] = self.L
#         # self._values['N'] = int(self.N)
       
#         #
        
#         # self._values['x_post'] = update
#         # self._values['x_prior'] = self.enkf_model.x_post
#         # self._values['factor']= self._values['x_post']//self._values['x_prior'] # round up factor values to int comment round it up
        
#         #
#         # self._values['dim_x'] = self.dim_x
#         #self.x, self.P, self.dim_z,self.dt, self.N,self.hx, self.fx
        
    def scale_output(self):

        self._values['x'] = self.x
        self._values['P'] = self.P
        self._values['dt'] = self.dt
        self._values['z'] = self.z
        self._values['N'] = self.N
        self._values['basin_area_km2']=self.basin_area_km2
        # self._values['validity'] = self.validity
        #self._values['site'] = self.total_discharge
    #________________________________________________________
    
    #________________________________________________________
    def config_from_json(self):
        with open(self.cfg_file) as data_file:
            data_loaded = json.load(data_file)
        # MANDATORY CONFIGURATIONS
        self.dt                 = data_loaded['time_step_in_seconds']
        self.x                  =np.array([float(data_loaded['initial_mean'])])
        self.P                 =np.eye(1)*float(data_loaded['initial_cov'])
        self.z                 =np.array([float(data_loaded['initial_obs'])])
        self.F           =float(data_loaded['state_transition_multiplier_matrix'])
        self.N           =data_loaded['Number_of_ensembles']
        self.L           = np.array(data_loaded['look_up_table'])
        self.max_soil_res_def =data_loaded['max_state_var_change_soilResDef']
        self.basin_area_km2=data_loaded['basin_area_km2']
        return


    #-------------------------------------------------------------------
    # BMI: Model Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
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

