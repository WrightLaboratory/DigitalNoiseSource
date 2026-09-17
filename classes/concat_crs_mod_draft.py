## CHANGELOG:
## 02/25 Synchronization tests were successful, function added to synchronize data from drone and correlator
## 03/01 yaml config file tests were successful, need to write config saving script
## 03/08 Writing yaml config files to speed up loading times and reuse parameters found via iterative fitting...
## 03/09 Writing yaml interpreter that searches for the existing params when an init is called
## 07/30 Generalization pass:
##   - Synchronization_Function() now actually applies t_drone_offset before/during
##     its coarse+fine time-fitting search (previously it silently ignored it and
##     only ever applied the freshly-fit t_delta_dji, dropping any known/previously
##     established drone clock offset). self.t_drone_offset is now stored on the
##     instance so every internal re-synchronization uses the same offset.
##   - __init__ now crops to t_bounds BEFORE any interpolation/processing happens
##     (previously the crop was applied only at the very end, after already doing
##     all of the interpolation work), and does so in the order: apply
##     t_drone_offset to the drone timestamps first, then crop by t_bounds. Each
##     function below still accepts its own t_bounds for extra cropping later.
##   - Fixed a bug where self.t_index kept its *original* (pre-crop) absolute
##     values after slicing, while every other time-axis array was re-indexed to
##     be 0-based -- this silently broke any downstream indexing (e.g.
##     fitting_utils.Fit_Main_Beam) whenever t_bounds was used.
##   - Fixed a bug where self.V_cross was checked with hasattr() (always True,
##     since the attribute is always assigned, even when None) instead of an
##     explicit `is not None` check; datasets with no cross-correlations no
##     longer crash on the t_bounds crop.
##   - Removed stray unconditional debug print()s that ignored self.traceback.
##   - Generalized functions to work for any number of channels (not just even
##     counts, and not assuming every channel comes paired with an X/Y partner
##     channel on the same dish), and to gracefully skip channels with no
##     associated dish (NaN dish_coords) instead of crashing.
##   - Synchronization_Function() now saves the main-beam fit parameters (Airy +
##     2DGauss) found at the final synchronized time offset as
##     self.sync_A_popt/self.sync_A_PR/self.sync_G_popt/self.sync_G_PR (plus
##     self.sync_chans/self.sync_freqs), and writes them to disk as an .npz file
##     when save_traceback=True, so FWHM-vs-frequency (etc.) plots can be made
##     later without re-running the whole synchronization search.
## 07/30 (2) Synchronization_Function() gained an apply_t_delta_dji=False option.
##     t_drone_offset is always used as the search baseline; t_delta_dji is
##     always found/evaluated/saved either way, but by default it no longer
##     gets applied to self's own drone_*_interp coordinates -- pass
##     apply_t_delta_dji=True once you trust the fit to actually shift them.
##     The verification plots and saved fit params now always reflect
##     t_delta_dji (via an internal tempconcat) regardless of this flag, so you
##     can judge the timing solution before committing to it.
## 08/19 Fixed _make_temp_concat() building its throwaway CONCAT copies from the
##     module-level `concat` import (concat.CONCAT(...)) instead of from this
##     class itself. That external `concat` module can drift out of sync with
##     this file (e.g. an older CONCAT.__init__ that predates the t_bounds
##     parameter added above), causing a confusing
##     "TypeError: __init__() got an unexpected keyword argument 't_bounds'"
##     even though the CONCAT actually running the search supports it fine.
##     _make_temp_concat now builds copies via type(self)(...) so it is always
##     self-consistent with whatever class is actually executing, regardless of
##     the state of any separately-imported concat module.
## 08/21 Unified auto/cross channel model (matches new corr.py): removed
##     self.crossmap/self.V_cross and the chan_type='auto'|'cross' mechanism
##     entirely. Every self.V[...,chan] is now already a fully-formed
##     correlation product (auto or cross); everything reads it uniformly via
##     np.abs(). Added self.chan_prod/self.chan_is_auto (passthrough from
##     corr.py) and self.chan_dish_coords (per-channel resolved coordinate:
##     the dish's own coords for an auto channel, midpoint of the two dishes'
##     coords for a cross channel -- correctly resolving physical port number
##     -> chmap position first, so this works for an arbitrary/dynamic chmap
##     subset, not just a contiguous one). Synchronization_Function's chans
##     argument is now always a flat product index (0..n_channels-1); chan_type
##     is gone from its signature. Perform_Background_Subtraction now does
##     subtraction in complex space (no more separate V_cross_bg/_bgsub); V_bg/
##     V_bgsub are complex-native. Extract_Source_Pulses now reads np.abs(V).
##     Distance_Compensation: auto channels are now correctly resolved (was
##     silently assuming chan index == dish index, which breaks once chans are
##     product-indexed); cross channels are explicitly left UNCOMPENSATED with
##     a printed flag, since combining two ports' distance tracks into one
##     compensation curve is a real design decision, not something to guess at
##     silently -- see that function's docstring.
## 09/16 Added an optional copol_angle_deg=0.0 argument to Synchronization_Function()
##     (and Main_Beam_Fitting()) for flights where the instrument's copolarization
##     axes are not aligned with the standard NS/EW local-Cartesian x/y frame (e.g.
##     a dish mounted with polarizations along NW/NE instead of N/E -- 45 degrees
##     off nominal). Previously the main-beam 2D Gaussian fit's xsig/ysig -- and
##     therefore every FWHM derived from them -- were always computed along plain
##     x/y, silently assuming those axes were the copol axes, which is wrong for a
##     rotated dish. copol_angle_deg=0.0 (the default) reproduces the exact old
##     behavior for every existing call; see fitting_utils.Fit_Main_Beam's
##     docstring/_rotate_xy() for what it does and the sign convention. Threaded
##     through to every internal Fit_Main_Beam call in Synchronization_Function
##     (coarse search, fine search, final saved sync fit) and to
##     Synchronization_Verification_Plots, and recorded as self.sync_copol_angle_deg
##     / in the saved .npz, so results are self-documenting later.
## 09/17 Two fixes to Main_Beam_Fitting()'s saved .npz, found while comparing it
##     against Synchronization_Function()'s:
##       - copol_angle_deg was accepted as an argument (per 09/16 above) but,
##         unlike Synchronization_Function()'s .npz, was never actually written
##         into Main_Beam_Fitting()'s saved file -- a file fit with a nonzero
##         rotation had no record of which frame its xsig/ysig were in. Now saved.
##       - Main_Beam_Fitting()'s .npz only ever had A_popt/A_PR/G_popt/G_PR, so
##         pu.Plot_Sync_FWHM_vs_Frequency() (which reads chan_indices/freq_MHz/
##         G_popt) could only ever be pointed at Synchronization_Function()'s
##         sparse synchronization-check-frequency file, not at a real full-band
##         fit. Main_Beam_Fitting() now also saves chan_indices/chan_prod/
##         freq_indices/freq_MHz, matching Synchronization_Function()'s schema,
##         so either file works interchangeably with that plotting function.
##     Also: Synchronization_Function() used to silently reuse a cached
##     t_delta_dji/copolchan when called again with a *different* chans array
##     than whatever produced the cache (only a caution message was printed) --
##     it now compares the requested chans against self.sync_chans (set once a
##     prior call actually completes) and invalidates/re-searches automatically
##     on a mismatch, so a stale timing solution can't get silently applied to
##     the wrong channels' FWHM fit.
## 09/17 (2) Synchronization_Function() and Main_Beam_Fitting() now also save
##     the fitted beam center explicitly, rather than leaving callers to
##     remember that G_popt[:,:,1]/[:,:,3] are x0/y0. Synchronization_Function
##     saves self.sync_x0/self.sync_y0 (per chan+freq) and self.sync_center_xy
##     (per-channel, nanmedian over frequency -- a more robust "where's the
##     beam actually centered" value than trusting any single per-frequency
##     fit, since the physical beam center shouldn't move with frequency but
##     any one per-frequency Gaussian fit can be noisy, e.g. at low SNR).
##     Main_Beam_Fitting saves the equivalent self.x0/self.y0/self.center_xy,
##     matching its existing chan_indices/freq_MHz/G_popt schema symmetry with
##     Synchronization_Function's .npz (see the first 09/17 entry above). Note
##     that when copol_angle_deg is nonzero, these x0/y0/center values are in
##     that same rotated (copol) frame as G_popt itself -- see
##     sync_copol_angle_deg / the saved copol_angle_deg field to know which
##     frame a given saved center is in. All new fields are written into the
##     saved .npz alongside the existing fit arrays.
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import hdf5plugin
import os
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz
import bisect
import pygeodesy
import yaml
from scipy.signal import square
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
## Import packages from our own module:
sys.path.insert(0, '../classes/')
import corr_crs_mod_draft as corr
import concat_crs_mod_draft as concat
import drone_crs_mod_draft as drone
import plotting_utils_crs_mod_draft as pu
import fitting_utils_crs_mod_draft as fu
import geometry_utils as gu
import time_utils as tu
import site_utils as si
## What could go wrong? ...
#from beamcals import concat
class CONCAT:
    def __init__(self,CORRDATCLASS,DRONEDATCLASS,config_directory="/hirax/GBO_Analysis_Outputs/concat_config_files/",output_directory='/hirax/GBO_Analysis_Outputs/',load_yaml=True,traceback=True,save_traceback=True,t_drone_offset=0,t_bounds=None):
        ## Decide whether or not we want a traceback for print statements/verification plots:
        self.traceback=traceback
        ## Decide whether or not we want to save traceback output plots:
        self.save_traceback=save_traceback
        self.t_bounds = t_bounds
        ## Store the drone timing offset so any downstream function (most
        ## notably Synchronization_Function) can reuse the exact same
        ## correction instead of silently defaulting to zero:
        self.t_drone_offset = t_drone_offset
        ## Import file information from both corr and drone classes:
        self.name=DRONEDATCLASS.name
        self.Data_Directory=CORRDATCLASS.Data_Directory
        self.Gain_Directory=CORRDATCLASS.Gain_Directory
        self.filenames=CORRDATCLASS.filenames
        self.gainfile=CORRDATCLASS.gainfile
        self.Drone_Directory=DRONEDATCLASS.Drone_Directory
        self.FLYTAG=DRONEDATCLASS.FLYTAG
        ## YAML configuration variables and config parameter loading:
        self.Config_Directory=config_directory
        self.load_yaml=load_yaml
        if self.traceback==True:
            print('Initializing CONCAT CLASS with active traceback using:')
            print("  --> "+CORRDATCLASS.Data_Directory)
            print("  --> "+DRONEDATCLASS.FLYTAG)
            if self.save_traceback==True:
                print('Creating directory for saving traceback and analysis outputs:')
                if 'TONE_ACQ' in self.Data_Directory:
                    tmpcorrdir = self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
                elif 'NFandFF' in self.Data_Directory:
                    tmpcorrdir = self.Data_Directory.split("_Suit")[0].split("NFandFF/")[1]
                else:
                    tmpcorrdir = os.path.basename(os.path.normpath(self.Data_Directory))
                tmpdronedir=self.FLYTAG.split('.')[0]
                tmpoutputdir=output_directory+'{}_{}'.format(tmpdronedir,tmpcorrdir)+'/'
                if os.path.exists(tmpoutputdir)==False:
                    self.Output_Directory=tmpoutputdir
                    self.Output_Prefix='{}_{}'.format(tmpdronedir,tmpcorrdir)
                if os.path.exists(tmpoutputdir)==True:
                    suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    self.Output_Prefix='{}_{}_ver_{}'.format(tmpdronedir,tmpcorrdir,suff)
                    self.Output_Directory=output_directory+'{}_{}_ver_{}'.format(tmpdronedir,tmpcorrdir,suff)+'/'
                os.makedirs(self.Output_Directory)
                print("  --> "+self.Output_Directory)
            if self.save_traceback==False:
                print("  --> Traceback outputs will not be saved...")
        elif self.traceback==False:
            pass
        ## If we don't want to load previous yaml config files, we skip the yaml interpreter:
        if self.load_yaml==False:
            if self.traceback==True:
                print('Concat initialized without previous config file...')
            if traceback==False:
                pass
        ## If we DO want to load previous config files, we enter this loop:
        if self.load_yaml==True:
            ## First we check if a config has previously been saved using these two data sets:
            if 'TONE_ACQ' in self.Data_Directory:
                tmpcorrdir=self.Data_Directory.split("_yale")[0].split("TONE_ACQ/")[1]
            elif 'NFandFF' in self.Data_Directory:
                tmpcorrdir=self.Data_Directory.split("_Suit")[0].split("NFandFF/")[1]
            tmpdronedir=self.FLYTAG.split('.')[0]
            tmpconfigpath=self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir)
            self.yaml_exists=os.path.exists(tmpconfigpath)
            ## If the previous config file doesn't exist, we will tell the user and move on:
            if self.yaml_exists==False:
                ## Check to see if we are going to use some print statements:
                if self.traceback==True:
                    print('Searching for previous config file:')
                    print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
                    print('    --> FILE NOT FOUND')
                    print('    --> Parameters can not be imported...')
                if self.traceback==False:
                    pass
            ## If the previous config file exists, we enter this loop to load parameters:
            if self.yaml_exists==True:
                ymlfile=open(self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir))
                #documents=yaml.full_load(ymlfile)
                with open(ymlfile, 'r') as fff:
                    documents = yaml.safe_load(fff)
                ## Check to see if the variables exist for function: Extract_Source_Pulses
                if 'pulse_params' in documents.keys():
                    self.pulse_period=np.float64(documents["pulse_params"]["pulse_period"])
                    self.pulse_dutycycle=np.float64(documents["pulse_params"]["pulse_dutycycle"])
                    self.t_delta_pulse=np.float64(documents["pulse_params"]["t_delta_pulse"])
                ## Check to see if the variables exist for function: Synchronization_Function
                if 'timing_params' in documents.keys():
                    self.t_delta_dji=np.float64(documents["timing_params"]["t_delta_dji"])
                    self.copolchan=documents["timing_params"]["copolchan"]
                ymlfile.close() # close the config file
                if self.traceback==True:
                    print('Searching for previous config file:')
                    print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
                    print('    --> FILE EXISTS')
                    print('    --> Loading existing config file parameters...')
                    if hasattr(self,"pulse_period")==True:
                        print('      --> pulse_period = {}'.format(self.pulse_period))
                    if hasattr(self,"pulse_dutycycle")==True:
                        print('      --> pulse_dutycycle = {}'.format(self.pulse_dutycycle))
                    if hasattr(self,"t_delta_pulse")==True:
                        print('      --> t_delta_pulse = {}'.format(self.t_delta_pulse))
                    if hasattr(self,"t_delta_dji")==True:
                        print('      --> t_delta_dji = {}'.format(self.t_delta_dji))
                    if hasattr(self,"copolchan")==True:
                        print('      --> copolchan = {}'.format(self.copolchan))
                if self.traceback==False:
                    pass
        ## Import vars from corr and drone classes:
        self.n_dishes=CORRDATCLASS.n_dishes
        self.n_channels=CORRDATCLASS.n_channels
        self.chmap=CORRDATCLASS.chmap
        self.automap=CORRDATCLASS.automap
        self.chan_prod=CORRDATCLASS.chan_prod
        self.chan_is_auto=CORRDATCLASS.chan_is_auto
        self.origin=DRONEDATCLASS.origin
        self.prime_origin=DRONEDATCLASS.prime_origin
        self.dish_keystrings=DRONEDATCLASS.dish_keystrings
        self.dish_coords=DRONEDATCLASS.dish_coords
        self.dish_pointings=DRONEDATCLASS.dish_pointings
        self.dish_polarizations=DRONEDATCLASS.dish_polarizations
        ## Every saved channel is already a fully-formed correlation product
        ## (auto or cross) -- there's no "raw per-port" array anymore, so each
        ## channel's own coordinate is resolved once, here. chan_prod holds
        ## *physical port numbers* (straight from the file's index_map/prod),
        ## while dish_coords/chmap are ordered by *position within chmap* --
        ## so a port number must be resolved to its chmap position before
        ## indexing dish_coords. This works whether chmap is a contiguous
        ## 0..N-1 range or an arbitrary dynamic subset (e.g. iceboard):
        self._port_to_dish_idx={int(p):idx for idx,p in enumerate(self.chmap)}
        self.chan_dish_coords=np.nan*np.ones((self.n_channels,self.dish_coords.shape[1]))
        for k in range(self.n_channels):
            pa,pb=self.chan_prod[k]
            ia=self._port_to_dish_idx.get(int(pa))
            ib=self._port_to_dish_idx.get(int(pb))
            if ia is None or ib is None:
                continue
            if self.chan_is_auto[k]:
                self.chan_dish_coords[k]=self.dish_coords[ia]
            else:
                self.chan_dish_coords[k]=np.nanmean(np.vstack([self.dish_coords[ia],self.dish_coords[ib]]),axis=0)
        ## Time dimensions of all arrays must be concat with receiver data. Time index is therefore defined wrt telescope data.
        self.freq=CORRDATCLASS.freq
        self.t=CORRDATCLASS.t
        self.t_index=CORRDATCLASS.t_index
        self.t_arr_datetime=CORRDATCLASS.t_arr_datetime
        self.V=CORRDATCLASS.V
        ## STEP 1: crop the correlator-side data to any user-specified start/stop
        ## time indices FIRST, before any interpolation or further processing is
        ## done. (t_drone_offset only concerns drone<->correlator alignment, so it
        ## has no bearing on this step -- it is applied next, to the drone
        ## timestamps, before they get interpolated onto this now-cropped grid.)
        if self.t_bounds is not None:
            i0, i1 = self.t_bounds
            self.t_arr_datetime = self.t_arr_datetime[i0:i1]
            self.t = self.t[i0:i1]
            self.V = self.V[i0:i1, :, :]
            ## t_index must stay 0-based / positional so that every other
            ## function (which uses t_index values to directly index into V,
            ## drone_xyz_LC_interp, etc.) keeps working after the crop:
            self.t_index = np.arange(len(self.t_arr_datetime))
        ## STEP 2: apply the drone timing correction to the drone timestamps
        ## before doing anything else with them (interpolation, windowing, etc):
        drone_t_arr_datetime = DRONEDATCLASS.t_arr_datetime + datetime.timedelta(seconds=t_drone_offset)
        drone_t_min = drone_t_arr_datetime[0]
        drone_t_max = drone_t_arr_datetime[-1]
        ## Define lb and ub t_index corresponding to (offset-corrected) drone data start/stop times:
        CORR_t_ind_lb = bisect.bisect_right(self.t_arr_datetime, drone_t_min)
        CORR_t_ind_ub = bisect.bisect_left(self.t_arr_datetime, drone_t_max)
        ## Define interpolation time vectors for drone and corr data:
        tsepoch=datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.UTC)
        ds_CORR=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in self.t_arr_datetime[CORR_t_ind_lb:CORR_t_ind_ub]])
        ds_drone=np.array([(np.datetime64(ts).astype(datetime.datetime).replace(tzinfo=pytz.UTC)-tsepoch).total_seconds() for ts in drone_t_arr_datetime])
        if self.traceback==True:
            print("Interpolating drone coordinates for each correlator timestamp:")
            print("  --> correlator timestamp axis contains {} elements".format(len(ds_CORR)))
            print("  --> drone timestamp axis contains {} elements".format(len(ds_drone)))
        elif self.traceback==False:
            pass
        ## Create useful drone coordinate arrays which we must interp, NAN non valued elements:
        self.drone_llh_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_xyz_LC_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_rpt_interp=np.nan*np.ones((self.t_arr_datetime.shape[0],3))
        self.drone_yaw_interp=np.nan*np.ones(self.t_arr_datetime.shape[0])
        self.drone_pitch_interp=np.nan*np.ones(self.t_arr_datetime.shape[0])
        self.drone_xyz_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.xyz_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        self.drone_rpt_r_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        self.drone_rpt_t_per_dish_interp=np.nan*np.ones((DRONEDATCLASS.rpt_r_per_dish.shape[0],self.t_arr_datetime.shape[0],3))
        ## Interp Drone variables:
        for i in [0,1,2]:
            self.drone_llh_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_llh[:,i])
            self.drone_xyz_LC_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_xyz_LC[:,i])
            self.drone_rpt_interp[CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.coords_rpt[:,i])
            for j in range(DRONEDATCLASS.rpt_r_per_dish.shape[0]):
                self.drone_xyz_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.xyz_per_dish[j,:,i])
                self.drone_rpt_r_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_r_per_dish[j,:,i])
                self.drone_rpt_t_per_dish_interp[j,CORR_t_ind_lb:CORR_t_ind_ub,i]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.rpt_t_per_dish[j,:,i])
        self.drone_yaw_interp[CORR_t_ind_lb:CORR_t_ind_ub]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.yaw[:])
        self.drone_pitch_interp[CORR_t_ind_lb:CORR_t_ind_ub]=np.interp(ds_CORR,ds_drone,DRONEDATCLASS.pitch[:])
        self.tstep=1e-9*np.nanmedian(np.diff(self.t))
    def Extract_Source_Pulses(self,Period=0.4e6,Dutycycle=0.2e6,t_bounds=[0,-1],f_ind=900,
                          minmaxpercents=[10.0,99.5],use_ref_channel=None):
        ## Search for all three timing variables that must be loaded from config:
        if hasattr(self,"pulse_period")==True and hasattr(self,"pulse_dutycycle")==True and hasattr(self,"t_delta_pulse")==True:
            if self.traceback==True:
                print("Extracting Source Pulses using parameters loaded from config file:")
                print('  --> pulse_period = {}'.format(self.pulse_period))
                print('  --> pulse_dutycycle = {}'.format(self.pulse_dutycycle))
                print('  --> t_delta_pulse = {}'.format(self.t_delta_pulse))
            if self.traceback==False:
                pass
            concat_duration=int(np.ceil((self.t_arr_datetime[-1]-self.t_arr_datetime[0]).total_seconds()))
            time_s,time_dt,switch=tu.Pulsed_Data_Waveform(total_duration=concat_duration,
                                                          period=self.pulse_period,
                                                          duty_cycle_on=self.pulse_dutycycle)
        ## If we don't have any variables, then we haven't loaded a yaml yet... and must run the function:
        if hasattr(self,"t_delta_pulse")==False:
            ## Create Switch Signal
            self.pulse_period=Period
            self.pulse_dutycycle=Dutycycle
            concat_duration=int(np.ceil((self.t_arr_datetime[-1]-self.t_arr_datetime[0]).total_seconds()))
            time_s,time_dt,switch=tu.Pulsed_Data_Waveform(total_duration=concat_duration,
                                                          period=self.pulse_period,
                                                          duty_cycle_on=self.pulse_dutycycle)
            ## Create t_offset range (1 period) and Pearson_r vars:
            t_offset_dist=np.arange(-1.0*self.pulse_period*1e-6,0.0,0.001)
            Pr_arr=np.NaN*np.ones((self.n_channels,t_offset_dist.shape[0]))
            Pr_max_ind_per_channel=np.NaN*np.ones(self.n_channels)
            Pr_max_t_0_per_channel=np.NaN*np.ones(self.n_channels)
            t_full=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime[:]])
            ## Loop over channels to find/plot a time offset solution with some clever fitting:
            if self.traceback==True:
                fig1,ax1=subplots(nrows=1,ncols=1,figsize=(16,4))
            for i in range(self.n_channels):
                ## Every channel is already a formed correlation product (auto
                ## or cross) -- np.abs() is the uniform, correct way to get its
                ## magnitude regardless of which type it is:
                V_mag_i=np.abs(self.V[:,f_ind,i])
                minsubdata=V_mag_i-np.percentile(V_mag_i,minmaxpercents[0])
                normminsubdata=minsubdata/np.percentile(minsubdata,minmaxpercents[1])
                clipnormminsubdata=normminsubdata.clip(0,1)
                stepped_func=interp1d(t_full,clipnormminsubdata,kind='previous',fill_value='extrapolate')
                sniparr=np.where(time_s[np.where(time_s<=t_full[t_bounds[1]])[0]]>=t_full[t_bounds[0]])[0]
                t_restrict=np.intersect1d(np.arange(len(time_s))[~np.isnan(stepped_func(time_s))],sniparr)
                ## Loop over all time offsets in t_offset_dist to find maximum correlation:
                for j,t_offset in enumerate(t_offset_dist):
                    shiftedswitch=np.interp(time_s,time_s+t_offset,switch)
                    try:
                        Pr_arr[i,j]=pearsonr(stepped_func(time_s[t_restrict]),shiftedswitch[t_restrict])[0]
                    except ValueError:
                        Pr_arr[i,j]=np.nan
                if self.traceback==True:
                    ax1.plot(t_offset_dist,Pr_arr[i,:],'.')
                try:
                    maxPrind=np.where(Pr_arr[i,:]==np.nanmax(Pr_arr[i,:]))[0][0]
                    if self.traceback==True:
                        ax1.plot(t_offset_dist[maxPrind],Pr_arr[i,maxPrind],'ro')
                    Pr_max_ind_per_channel[i]=maxPrind
                    Pr_max_t_0_per_channel[i]=t_offset_dist[maxPrind]
                except IndexError:
                    Pr_max_ind_per_channel[i]=np.nan
                    Pr_max_t_0_per_channel[i]=np.nan
                if self.traceback==True:
                    print('  --> Pearson R offset solution, channel {}: {}'.format(i, Pr_max_t_0_per_channel[i]))
            ## --- Selection of t_delta_pulse depending on use_ref_channel ---
            if use_ref_channel is None:
                # Default: median of all channels
                self.t_delta_pulse=np.nanmedian(Pr_max_t_0_per_channel)
                chosen_channel = None
            else:
                # Use explicitly chosen channel
                chosen_channel = use_ref_channel
                self.t_delta_pulse = Pr_max_t_0_per_channel[chosen_channel]
            if self.traceback==True:
                ax1.axvline(self.t_delta_pulse,label="selected t_offset")
                ax1.legend(loc=1)
                print("Maximum Pearson_R Correlations between data and square wave function:")
                print("  --> t_indices = {}".format(Pr_max_ind_per_channel))
                print("  --> t_deltas = {}".format(np.around(Pr_max_t_0_per_channel,decimals=3)))
                if chosen_channel is None:
                    print("Selecting square wave function time offset (median of channels):")
                else:
                    print("Selecting square wave function time offset (forced channel {}):".format(chosen_channel))
                print("  --> t_delta_pulse = {:.10f}".format(self.t_delta_pulse))
                if self.save_traceback==True:
                    savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_pulse_Pearson_R.png")
        ## Interpolate the switching function with the concat timestamps:
        t_for_interp_out=np.array([(m-self.t_arr_datetime[0]).total_seconds() for m in self.t_arr_datetime[:]])
        t_for_interp_in=np.array([m.total_seconds() for m in time_dt])
        switch_interp_f=np.interp(t_for_interp_out,t_for_interp_in+self.t_delta_pulse,switch)
        self.switch_signal=switch
        self.switch_time=t_for_interp_in
        self.switch_signal_interp=switch_interp_f
        ## Once we have our time offset, we must extract indices where the source is on/off/rising:
        self.inds_span=np.union1d(list(set(np.where(np.diff(np.sign(switch_interp_f-0.5)))[0])),\
                                  np.intersect1d(np.where(1.0>switch_interp_f),np.where(switch_interp_f>0.0))).tolist()
        self.inds_on=list(set(np.where(switch_interp_f==1.0)[0])-set(self.inds_span))
        self.inds_off=list(set(np.where(switch_interp_f==0.0)[0])-set(self.inds_span))
        ## Each of these lists of indices should also have no overlap. Let's print to see:
        if self.traceback==True:
            print("Finding relevant pulsing indices and checking for overlaps:")
            print("  --> on/off ind intersection:",np.intersect1d(self.inds_on,self.inds_off))
            print("  --> on/span ind intersection:",np.intersect1d(self.inds_on,self.inds_span))
            print("  --> off/span ind intersection:",np.intersect1d(self.inds_off,self.inds_span))
            ## Let's plot the on/off/rising index groups. Grid sized to fit any
            ## number of channels (not just even counts):
            nrows=int(np.ceil(self.n_channels/2.0))
            fig3=figure(figsize=(16,int(4*nrows)))
            for i in range(self.n_channels):
                ax=fig3.add_subplot(nrows,2,i+1)
                V_mag=np.abs(self.V[:,f_ind,i])
                ax.semilogy(self.t_arr_datetime[:],V_mag,'k.',label='all')
                ax.semilogy(self.t_arr_datetime[self.inds_on],V_mag[self.inds_on],'.',label='on')
                ax.semilogy(self.t_arr_datetime[self.inds_off],V_mag[self.inds_off],'.',label='off')
                ax.semilogy(self.t_arr_datetime[self.inds_span],V_mag[self.inds_span],'x',label='span')
                ax.semilogy(self.t_arr_datetime[:],(np.nanmax(V_mag[self.inds_on])*switch_interp_f)+np.nanmin(V_mag[self.inds_on]),'--',alpha=0.1,label='switch, t_offset={:.2f}'.format(self.t_delta_pulse))
                ax.set_ylabel("Log Power Received [$ADU^2$]")
                ax.set_xlabel("Datetime")
                pa,pb=self.chan_prod[i]
                chlabel='Auto Ch {}'.format(pa) if self.chan_is_auto[i] else 'Cross {}x{}'.format(pa,pb)
                ax.set_title(chlabel)
                ax.legend(loc=2)
                ax.set_xlim(self.t_arr_datetime[t_bounds[0]],self.t_arr_datetime[t_bounds[1]])
            tight_layout()
            if self.save_traceback==True:
                savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_pulse_index_solution.png")
            if self.save_traceback==False:
                pass
        if self.traceback==False:
            pass
    def Perform_Background_Subtraction(self,window_size=5,t_bounds=None):
        ## BACKGROUND SUBTRACTED SPECTRA: ##
        if self.traceback==True:
            print("Calculating background spectra from indices where the noise source is off.")
        if self.traceback==False:
            pass
        if t_bounds is None:
            t_bounds=[0,-1]
        ## Restrict which time samples get (re)computed to any extra t_bounds
        ## crop requested here, on top of whatever is already in self.t_index:
        loop_t_index=self.t_index[t_bounds[0]:t_bounds[1]]
        self.V_bg=np.zeros(self.V.shape).astype(complex)
        self.V_bgsub=np.zeros(self.V.shape).astype(complex)
        ## Loop over all indices (within t_bounds) and construct the V_bg array:
        for k in loop_t_index:
            ## If ind is an off spectra, use this off spectra for the background:
            if k in self.inds_off:
                self.V_bg[k,:,:]=self.V[k,:,:]
            ## If ind is an on spectra, create an off spectra by averaging the before/after off spectra:
            elif k in np.union1d(self.inds_on,self.inds_span):
                t_window=np.intersect1d(np.arange(k-window_size,k+window_size),self.inds_off)
                self.V_bg[k,:,:]=np.nanmean(self.V[t_window,:,:],axis=0)
        ## V is complex-native (auto channels carry ~0 imaginary part, cross
        ## channels genuinely don't) -- subtraction happens here in complex
        ## space, BEFORE any magnitude is taken. Downstream consumers (e.g.
        ## Fit_Main_Beam via Vargs='bgsub') take np.abs() themselves when they
        ## actually read this array:
        self.V_bgsub=self.V-self.V_bg
        if self.traceback==True:
            print("  --> Background subtraction completed using window_size = {}".format(window_size))
        if self.traceback==False:
            pass
    def Synchronization_Function(self,inputcorr,inputdrone,coarse_params=[-10.0,10.0,0.2],fine_params=[-0.5,0.5,0.01],chans=np.array([0,1]),freqs=np.arange(100,1024,150),FMB_coordbounds=[30.0,30.0,150.0],FMB_ampbound=0.999,t_bounds=None,apply_t_delta_dji=False,copol_angle_deg=0.0):
        """
        copol_angle_deg : float, default 0.0
            Rotation (degrees) of this instrument's true copolarization axes
            away from the standard NS/EW local-Cartesian x/y frame -- see
            fitting_utils.Fit_Main_Beam's docstring (and _rotate_xy) for the
            exact convention. Every flight so far has had copol axes aligned
            with x/y, so 0.0 (unchanged behavior) stays the default. For a
            dish mounted with its polarizations along NW/NE instead of N/E,
            pass copol_angle_deg=45.0 (or -45.0 -- check sign against the
            verification plots this saves/shows). This is threaded through
            to every internal fu.Fit_Main_Beam call (coarse search, fine
            search, and the final saved sync fit) and to
            pu.Synchronization_Verification_Plots, so the coarse/fine time
            search itself is also run against the correct copol axes, not
            just the final saved FWHM. It's stored alongside the other
            synchronization results as self.sync_copol_angle_deg.
        apply_t_delta_dji : bool, default False
            t_drone_offset (as set when this CONCAT object was constructed) is
            always applied/used as the baseline for the search below. t_delta_dji
            -- the additional timing correction this function finds (or loads
            from an existing config) -- is always computed/reported and used to
            evaluate + save + plot the fit quality, but by default is NOT applied
            to this object's own drone_*_interp coordinates. Pass
            apply_t_delta_dji=True once you're confident in the value (e.g. after
            reviewing sync_G_PR/sync_A_PR and the verification plots) to actually
            shift this object's coordinates by it. Since t_delta_dji is cached on
            self once found, re-calling with apply_t_delta_dji=True afterward is
            cheap -- it will not repeat the coarse/fine search.
        chans : array of int
            Flat channel indices (0..n_channels-1) into self.V / a
            tempconcat's .V. Each index is already a fully-formed correlation
            product -- auto (real, imag~0) or cross (complex, imag!=0) -- and
            is fit uniformly via np.abs(). Use self.chan_prod/self.chan_is_auto
            to look up which physical port(s) a given channel index
            corresponds to. All chans in one call are fit and compared against
            each other directly (copolchan selection), so mixing very
            different baseline types in one call may not be meaningful, but
            nothing here enforces that -- it's your call.
        """
        if self.traceback==True:
            print("Synchronizing data from correlator and drone:")
        if self.traceback==False:
            pass
        if t_bounds is None:
            t_bounds = self.t_bounds
        chans=np.asarray(chans)
        if np.any(chans>=self.n_channels):
            raise ValueError(
                "chans must index into 0..{} (n_channels, i.e. every saved auto+cross "
                "product), got {}.".format(self.n_channels-1,chans))
        if self.traceback==True:
            for c in chans:
                pa,pb=self.chan_prod[c]
                label='Auto Ch {}'.format(pa) if self.chan_is_auto[c] else 'Cross {}x{}'.format(pa,pb)
                print("  --> chans value {} = {} (fit via np.abs()).".format(c,label))
        ## Use the same drone timing offset that was applied when this CONCAT
        ## object was built. Previously this function ignored t_drone_offset
        ## entirely and only ever applied the freshly-fit t_delta_dji, silently
        ## dropping any known/previously-established drone clock offset before
        ## running its own time-fitting search:
        t_drone_offset = self.t_drone_offset
        def _chan_label(c):
            """Human-readable label for chans[c], used in prints/plot titles."""
            pa,pb=self.chan_prod[c]
            return 'Auto Ch {}'.format(pa) if self.chan_is_auto[c] else 'Cross {}x{}'.format(pa,pb)
        def _make_temp_concat(offset_seconds):
            """
            Build a throwaway CONCAT at a trial drone-timing offset. Every
            channel (auto or cross) is already fully resolved by __init__ --
            tc.V and tc.chan_dish_coords are correct for any chan the instant
            this returns, so no chan-type-specific post-processing is needed
            here (unlike the old crossmap-position scheme).

            Builds the temp copy via type(self)(...) rather than going through
            the module-level `concat` import -- that external module can be a
            different/stale version of this class (e.g. missing newer
            constructor kwargs like t_bounds), which would silently break this
            helper even though the CONCAT actually running this search is
            perfectly fine. type(self) guarantees we always build from the
            exact class this method is defined on.
            """
            origtaxis_local=inputdrone.t_arr_datetime[:]
            inputdrone.t_arr_datetime=origtaxis_local+datetime.timedelta(seconds=offset_seconds)
            tc=type(self)(
                CORRDATCLASS=inputcorr,
                DRONEDATCLASS=inputdrone,
                load_yaml=False,
                traceback=False,
                save_traceback=False,
                t_drone_offset=t_drone_offset,
                t_bounds=t_bounds)
            try:
                tc.inds_on=self.inds_on
            except AttributeError:
                pass
            inputdrone.t_arr_datetime=origtaxis_local
            return tc
        ## If a previous Synchronization_Function() call ON THIS OBJECT already
        ## cached t_delta_dji for a *different* chans combination, silently
        ## reusing it here would apply a timing solution/copolchan label fit
        ## for the wrong channels to this call's chans. self.sync_chans (only
        ## set once a run of this function actually completes, below) is the
        ## reliable record of what chans a cached t_delta_dji was fit for --
        ## a t_delta_dji loaded fresh from a yaml config instead has no
        ## sync_chans yet and is fine to use as-is. So: invalidate the cache
        ## and force a fresh coarse+fine search whenever this object's own
        ## previously-fit chans don't match what's being asked for now,
        ## rather than reusing a stale solution with just a caution message:
        if hasattr(self,"t_delta_dji") and hasattr(self,"sync_chans") and not np.array_equal(np.asarray(chans),self.sync_chans):
            if self.traceback==True:
                print("  --> Cached t_delta_dji was fit using chans={} but this call requested chans={} -- invalidating cache and re-running the coarse+fine search.".format(self.sync_chans,chans))
            del self.t_delta_dji
            del self.copolchan
        ## Begin by checking if the t_delta_dji parameter was already loaded from the config file:
        if hasattr(self,"t_delta_dji")==True:
            if self.traceback==True:
                print('  --> Loading parameter from existing configuration file: t_delta_dji = {}'.format(self.t_delta_dji))
                ## self.copolchan is a *position within chans* (not a raw channel
                ## value), fixed at whatever chans produced it -- if this config
                ## was cached from a different chans combination than the
                ## current call, this label (and the cached t_delta_dji itself)
                ## may not mean what you think. Falls back to the raw index if
                ## it's out of range for the current chans:
                copolchan_i=int(self.copolchan)
                if copolchan_i<len(chans):
                    print("Applying a time correction of {:.2f} seconds using {} fits (cached copolchan index {} -- verify this matches the chans you meant).".format(self.t_delta_dji,_chan_label(chans[copolchan_i]),copolchan_i))
                else:
                    print("Applying a time correction of {:.2f} seconds using cached copolchan index {} (out of range for current chans={} -- this cache was likely built with different chans; treat with caution).".format(self.t_delta_dji,copolchan_i,chans))
            if self.traceback==False:
                pass
        if hasattr(self,"t_delta_dji")==False:
            if self.traceback==True:
                print("  --> Previous t_delta_dji not found")
                print("  --> Calculating via 2DGauss fitting routine:")
                print("  --> Applying known t_drone_offset = {:.3f}s before searching for t_delta_dji".format(t_drone_offset))
            if self.traceback==False:
                pass
            ## Begin with specifying time axis for iteration:
            t_coarse=np.arange(coarse_params[0],coarse_params[1],coarse_params[2])
            ## Define output products from fits:
            AFit_f_params=np.zeros((len(t_coarse),len(chans),len(freqs),5))
            APRarr=np.zeros((len(t_coarse),len(chans),len(freqs)))
            GFit_f_params=np.zeros((len(t_coarse),len(chans),len(freqs),7))
            GPRarr=np.zeros((len(t_coarse),len(chans),len(freqs)))
            ## Begin iterative loop for coarse time axis:
            from tqdm.auto import tqdm
            for i, ttry in enumerate(tqdm(t_coarse, desc="Coarse search")):
                tempconcat=_make_temp_concat(ttry)
                ## Run fits:
                result=fu.Fit_Main_Beam(
                    tempconcat,
                    chans,
                    freqs,
                    theta_solve=False,
                    coordbounds=FMB_coordbounds,
                    ampbound=FMB_ampbound,
                    copol_angle_deg=copol_angle_deg)
                AFit_f_params[i]=result[0]
                APRarr[i]=result[1]
                GFit_f_params[i]=result[2]
                GPRarr[i]=result[3]
            ## Loop over the channels and frequencies to find the index that maximizes the Pearson R Array:
            GPRmax=np.zeros((len(chans),len(freqs)))
            GPRval=np.zeros((len(chans),len(freqs)))
            for i in range(len(chans)):
                for j in range(len(freqs)):
                    try:
                        GPRmax[i,j]=np.where(GPRarr[:,i,j]==np.nanmax(GPRarr[:,i,j]))[0][0]
                        GPRval[i,j]=GPRarr[int(GPRmax[i,j]),i,j]
                    except IndexError:
                        GPRmax[i,j]=np.nan
                        GPRval[i,j]=np.nan
            copolchan=np.where(np.nanmean(GPRval,axis=1)==np.nanmax(np.nanmean(GPRval,axis=1)))[0][0]
            ## Redefine time axis for fine resolution:
            tfine0=t_coarse[int(np.nanmedian(GPRmax[copolchan,:]))]
            t_fine=np.arange(tfine0+fine_params[0],tfine0+fine_params[1],fine_params[2])
            ## Define output products from fine resolution fits:
            AFit_f_params_fine=np.zeros((len(t_fine),len(chans),len(freqs),5))
            APRarr_fine=np.zeros((len(t_fine),len(chans),len(freqs)))
            GFit_f_params_fine=np.zeros((len(t_fine),len(chans),len(freqs),7))
            GPRarr_fine=np.zeros((len(t_fine),len(chans),len(freqs)))
            ## Begin iterative loop for coarse time axis:
            for i, ttry in enumerate(tqdm(t_fine, desc="Fine search")):
                tempconcat=_make_temp_concat(ttry)
                ## Run fits:
                result=fu.Fit_Main_Beam(tempconcat,chans,freqs,theta_solve=False,coordbounds=FMB_coordbounds,ampbound=FMB_ampbound,copol_angle_deg=copol_angle_deg)
                AFit_f_params_fine[i]=result[0]
                APRarr_fine[i]=result[1]
                GFit_f_params_fine[i]=result[2]
                GPRarr_fine[i]=result[3]
            ## Find the maximal Pearson R values for the fine time axis:
            GPRmax_fine=np.zeros((len(chans),len(freqs)))
            GPRval_fine=np.zeros((len(chans),len(freqs)))
            for i in range(len(chans)):
                for j in range(len(freqs)):
                    try:
                        GPRmax_fine[i,j]=np.where(GPRarr_fine[:,i,j]==np.nanmax(GPRarr_fine[:,i,j]))[0][0]
                        GPRval_fine[i,j]=GPRarr_fine[int(GPRmax_fine[i,j]),i,j]
                    except IndexError:
                        GPRmax_fine[i,j]=np.nan
                        GPRval_fine[i,j]=np.nan
            ## Which channel has the best gaussianity, and what time offset does that channel suggest?
            copolchan_fine=np.where(np.nanmean(GPRval_fine,axis=1)==np.nanmax(np.nanmean(GPRval_fine,axis=1)))[0][0]
            self.t_delta_dji=t_fine[int(np.nanmedian(GPRmax_fine[copolchan_fine,:]))]
            self.copolchan=copolchan_fine
            ## Now make plots if desired:
            if self.traceback==True:
                print("Applying a time correction of {:.2f} seconds using {} fits.".format(self.t_delta_dji,_chan_label(chans[self.copolchan])))
                fig0,[[ax1,ax2],[ax3,ax4]]=subplots(nrows=2,ncols=2,figsize=(16,12))
                coarse_axes = [ax1, ax2]
                for i in range(len(chans)):
                    ax = coarse_axes[i]
                    for k, find in enumerate(freqs):
                        ax.plot(t_coarse, GPRarr[:,i,k], label='{:.2f}MHz'.format(tempconcat.freq[find]))
                        ## Some (channel, freq) combinations can have NO valid
                        ## Pearson R at any trial offset (e.g. a frequency with
                        ## no usable points after the amplitude/coordinate
                        ## cuts) -- GPRmax is correctly NaN for those, so skip
                        ## marking a "best point" rather than crashing on
                        ## int(nan):
                        if not np.isnan(GPRmax[i,k]):
                            ax.plot(t_coarse[int(GPRmax[i,k])], GPRarr[int(GPRmax[i,k]),i,k], 'r.')
                    med_ind=np.nanmedian(GPRmax[i]) if not np.all(np.isnan(GPRmax[i])) else np.nan
                    if not np.isnan(med_ind):
                        ax.axvline(
                            t_coarse[int(med_ind)],
                            c='r',
                            label='median t = {:.2f}'.format(t_coarse[int(med_ind)])
                        )
                    ax.set_title('{} Coarse Offset Correlation'.format(_chan_label(chans[i])))
                    ax.set_xlabel('$\Delta$t $[sec]$')
                    ax.set_ylabel('Pearson R Value')
                    ax.legend(loc=1, fontsize='small')
                fine_axes = [ax3, ax4]
                for i in range(len(chans)):
                    ax = fine_axes[i]
                    for k, find in enumerate(freqs):
                        ax.plot(t_fine, GPRarr_fine[:,i,k], label='{:.2f}MHz'.format(tempconcat.freq[find]))
                        ## Same NaN guard as the coarse-search plot above:
                        if not np.isnan(GPRmax_fine[i,k]):
                            ax.plot(t_fine[int(GPRmax_fine[i,k])], GPRarr_fine[int(GPRmax_fine[i,k]),i,k], 'r.')
                    med_ind_fine=np.nanmedian(GPRmax_fine[i]) if not np.all(np.isnan(GPRmax_fine[i])) else np.nan
                    if not np.isnan(med_ind_fine):
                        ax.axvline(
                            t_fine[int(med_ind_fine)],
                            c='r',
                            label='median t = {:.2f}'.format(t_fine[int(med_ind_fine)])
                        )
                    ax.set_title('{} Fine Offset Correlation'.format(_chan_label(chans[i])))
                    ax.set_xlabel('$\Delta$t $[sec]$')
                    ax.set_ylabel('Pearson R Value')
                    ax.legend(loc=1, fontsize='small')
                tight_layout()
                if self.save_traceback==True:
                    savefig(self.Output_Directory+self.Output_Prefix+"_t_delta_dji_Pearson_R.png")
                if self.save_traceback==False:
                    pass
            elif self.traceback==False:
                pass
        ## END of config interpretation loop!
        ## Build a tempconcat with t_delta_dji applied on top of the known
        ## t_drone_offset -- this is used below to evaluate/save/plot the
        ## quality of this timing solution REGARDLESS of apply_t_delta_dji,
        ## since the whole point of that flag is to let you inspect the
        ## solution before committing to it:
        tempconcat=_make_temp_concat(self.t_delta_dji)
        ## Only overwrite THIS object's drone coordinates with the
        ## t_delta_dji-shifted values if explicitly asked to. Otherwise self
        ## keeps exactly what __init__ gave it (t_drone_offset only) -- t_delta_dji
        ## is still found/reported/saved above, just not applied here:
        if apply_t_delta_dji:
            if self.traceback==True:
                print('  --> apply_t_delta_dji=True: applying t_delta_dji = {:.3f}s on top of t_drone_offset = {:.3f}s to this object\'s drone coordinates.'.format(self.t_delta_dji,t_drone_offset))
            self.drone_llh_interp=tempconcat.drone_llh_interp
            self.drone_xyz_LC_interp=tempconcat.drone_xyz_LC_interp
            self.drone_rpt_interp=tempconcat.drone_rpt_interp
            self.drone_yaw_interp=tempconcat.drone_yaw_interp
            self.drone_pitch_interp=tempconcat.drone_pitch_interp
            self.drone_xyz_per_dish_interp=tempconcat.drone_xyz_per_dish_interp
            self.drone_rpt_r_per_dish_interp=tempconcat.drone_rpt_r_per_dish_interp
            self.drone_rpt_t_per_dish_interp=tempconcat.drone_rpt_t_per_dish_interp
        else:
            if self.traceback==True:
                print('  --> apply_t_delta_dji=False: t_delta_dji = {:.3f}s found/loaded but NOT applied. This object\'s drone coordinates still only reflect t_drone_offset = {:.3f}s. Re-run with apply_t_delta_dji=True to apply it (this will not repeat the search, since t_delta_dji is already cached).'.format(self.t_delta_dji,t_drone_offset))
        ## Save the main-beam fit parameters at t_delta_dji so that later
        ## analyses/plots (e.g. FWHM vs frequency, or the diagnostic checks
        ## above) can be made without re-running the whole coarse+fine search.
        ## This always uses tempconcat (i.e. t_delta_dji IS evaluated here) so
        ## you can judge the fit quality even when apply_t_delta_dji=False:
        if self.traceback==True:
            print('  --> Fitting main beam at t_delta_dji={:.3f}s for evaluation/later use (e.g. FWHM vs frequency):'.format(self.t_delta_dji))
        sync_A_popt,sync_A_PR,sync_G_popt,sync_G_PR=fu.Fit_Main_Beam(
            tempconcat,chans,freqs,theta_solve=False,coordbounds=FMB_coordbounds,ampbound=FMB_ampbound,copol_angle_deg=copol_angle_deg)
        self.sync_chans=np.asarray(chans)
        ## Record which physical port(s) each sync_chans entry corresponds to,
        ## so results (and the .npz below) can be correctly interpreted later
        ## without needing to re-look-up self.chan_prod at that time:
        self.sync_chan_prod=self.chan_prod[self.sync_chans]
        self.sync_freqs=np.asarray(freqs)
        self.sync_A_popt=sync_A_popt
        self.sync_A_PR=sync_A_PR
        self.sync_G_popt=sync_G_popt
        self.sync_G_PR=sync_G_PR
        ## Record the copol rotation these fits were performed with, so a
        ## saved .npz (and G_popt's x0/y0/xsig/ysig) can be correctly
        ## interpreted later without having to remember what was passed in:
        self.sync_copol_angle_deg=copol_angle_deg
        ## Save the fitted beam center out explicitly, rather than leaving
        ## callers to remember that sync_G_popt[:,:,1]/[:,:,3] are x0/y0.
        ## self.sync_x0/self.sync_y0 are just that, pulled out under clear
        ## names (shape: [len(chans), len(freqs)]). self.sync_center_xy is
        ## additionally a single, more robust per-channel center (shape:
        ## [len(chans), 2]) -- the nanmedian of x0/y0 across every fit
        ## frequency for that channel. Use sync_center_xy as the go-to
        ## "where is the beam actually centered" value for plotting/analysis;
        ## sync_x0/sync_y0 remain available per-frequency for diagnosing a
        ## channel/frequency where the fit center looks off. Note these are
        ## in the same (possibly copol-rotated) frame as sync_G_popt itself --
        ## see self.sync_copol_angle_deg above.
        self.sync_x0=sync_G_popt[:,:,1]
        self.sync_y0=sync_G_popt[:,:,3]
        self.sync_center_xy=np.column_stack([
            np.nanmedian(self.sync_x0,axis=1),
            np.nanmedian(self.sync_y0,axis=1),
        ])
        if self.save_traceback==True:
            tmpsyncfitpath=self.Output_Directory+self.Output_Prefix+"_Synchronization_MainBeam_Fits.npz"
            np.savez(tmpsyncfitpath,
                     chan_indices=self.sync_chans,
                     chan_prod=self.sync_chan_prod,
                     freq_indices=self.sync_freqs,
                     freq_MHz=self.freq[self.sync_freqs],
                     A_popt=self.sync_A_popt,
                     A_PR=self.sync_A_PR,
                     G_popt=self.sync_G_popt,
                     G_PR=self.sync_G_PR,
                     x0=self.sync_x0,
                     y0=self.sync_y0,
                     center_xy=self.sync_center_xy,
                     t_delta_dji=self.t_delta_dji,
                     t_drone_offset=self.t_drone_offset,
                     copol_angle_deg=self.sync_copol_angle_deg,
                     applied=apply_t_delta_dji)
            if self.traceback==True:
                print('  --> Saved main beam fit parameters: {}'.format(tmpsyncfitpath))
        if self.traceback==True:
            print('  --> Generating output verification plots (always evaluated at t_delta_dji, regardless of apply_t_delta_dji):')
            channels=chans
            pu.Synchronization_Verification_Plots(inputconcat=tempconcat,chans=channels,find=freqs[-1],coordbounds=FMB_coordbounds,ampbound=FMB_ampbound,copol_angle_deg=copol_angle_deg)
            if self.save_traceback==True:
                print('  --> Saving output plot.')
                savefig(self.Output_Directory+self.Output_Prefix+"_Synchronization_Verification.png")
            if self.save_traceback==False:
                pass
        if self.traceback==False:
            pass
        ## This should result in reassigned drone coordinates (if apply_t_delta_dji=True), and eliminate a lot of computation time on repeat calls...
    def Export_yaml(self,):
        ## Create the file on disk at the specified path and write an initialization comment:
        tmpcorrdir=(self.Data_Directory.split("Z_")[0]+'Z').split("/")[-1]
        tmpdronedir=self.FLYTAG.split('.')[0]
        tmpconfigpath=self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir)
        if self.traceback==True:
            print('Preparing to export configuration file:')
            print('  --> Checking directory for config_{}_{}.yaml:'.format(tmpdronedir,tmpcorrdir))
            if os.path.exists(tmpconfigpath)==False:
                print('    --> FILE NOT FOUND')
                print('    --> preparing to write a new configuration file...')
            if os.path.exists(tmpconfigpath)==True:
                print('    --> FILE EXISTS')
                print('    --> preparing to write a new versioned configuration file...')
        if self.traceback==False:
            pass
        ## Create the configuration file at the specified location on disk:
        if os.path.exists(tmpconfigpath)==False:
            ymlfile=open(self.Config_Directory+'config_{}_{}.yaml'.format(tmpdronedir,tmpcorrdir), 'w')
        if os.path.exists(tmpconfigpath)==True:
            try:
                ymlfile=open(self.Config_Directory+'config_'+self.Output_Prefix+'.yaml', 'w')
            except AttributeError:
                suff=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                ymlfile=open(self.Config_Directory+'config_{}_{}_ver_{}.yaml'.format(tmpdronedir,tmpcorrdir,suff), 'w')
        ymlfile.write('#YAML_CONFIG_FILE for {} and {}\n'.format(tmpdronedir,tmpcorrdir))
        ## (0) Write comments that indicate whether or not certain functions worked on this data set:
        ymlfile.write('#LIST OF SUCCESFULLY RUN CONCAT CLASS FUNCTIONS:\n')
        if hasattr(self,'V')==True:
            ymlfile.write('# --> [X] concat.__init__()\n')
        if hasattr(self,'switch_signal_interp')==True:
            ymlfile.write('# --> [X] concat.Extract_Source_Pulses()\n')
        if hasattr(self,'V_bgsub')==True:
            ymlfile.write('# --> [X] concat.Perform_Background_Subtraction()\n')
        if hasattr(self,'t_delta_dji')==True:
            ymlfile.write('# --> [X] concat.Synchronization_Function()\n')
        ymlfile.write('\n') # new line
        ## (1) These file info parameters will by definition always exist:
        file_info={'file_info':{'name':self.name,
                                'Data_Directory':self.Data_Directory,
                                'filenames':self.filenames.astype(str).tolist(),
                                'Drone_Directory':self.Drone_Directory,
                                'FLYTAG':self.FLYTAG,
                                'Gain_Directory':self.Gain_Directory,
                                'gainfile':self.gainfile}}
        ## Write a comment about and insert the file info section into the .yaml:
        ymlfile.write('#FILENAMES AND PATHS ON DISK TO CONCATENATED FILES:\n')
        yaml.dump(file_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (2) These site geometry parameters will by definition always exist:
        site_info={'site_geometry_i':{'n_channels':self.n_channels,
                                    'n_dishes':self.n_dishes,
                                    'automap':self.automap.tolist(),
                                    'chmap':self.chmap.tolist(),
                                    'dish_keystrings':self.dish_keystrings.tolist(),
                                    'dish_coords':self.dish_coords.tolist(),
                                    'dish_pointings':self.dish_pointings.tolist(),
                                    'dish_polarizations':self.dish_polarizations.tolist()}}
        ## Write a comment about and insert the site info section into the .yaml:
        ymlfile.write('#INITIAL SITE GEOMETRY PARAMETERS AND VARIABLES USED:\n')
        yaml.dump(site_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (3) These correlator parameters will by definition always exist:
        corr_info={'corr_params':{'tstep':float(self.tstep),
                                  'fstep':float(np.nanmedian(np.diff(self.freq))),
                                  'dimensions':list(self.V.shape)}}
        ## Write a comment about and insert the site info section into the .yaml:
        ymlfile.write('#CORRELATOR PARAMETERS AND VARIABLES USED:\n')
        yaml.dump(corr_info, ymlfile, default_flow_style=None)
        ymlfile.write('\n') # new line
        ## (4) These pulse parameters will only exist if the source was pulsed and Extract_Source_Pulses() was run:
        try:
            pulse_info={'pulse_params':{'pulse_dutycycle':self.pulse_dutycycle,
                                        'pulse_period':self.pulse_period,
                                        't_delta_pulse':float(self.t_delta_pulse)}}
            ymlfile.write('#SOURCE PULSING PARAMETERS (microsec):\n')
            yaml.dump(pulse_info, ymlfile)
        except AttributeError:
            ymlfile.write('#SOURCE PULSING PARAMETERS NOT FOUND -- REQUIRES Extract_Source_Pulses()\n')
        ## Write a comment about and insert the pulse info section into the .yaml:
        ymlfile.write('\n') # new line
        ## (5) These timing parameters will only exist if Synchronization_Function was run:
        try:
            timing_info={'timing_params':{'t_delta_dji':float(self.t_delta_dji),
                                          'copolchan':float(self.copolchan),
                                          't_drone_offset':float(self.t_drone_offset)}}
            ymlfile.write('#BEST-FIT DRONE TIMESTAMP TIMING OFFSET (seconds):\n')
            yaml.dump(timing_info, ymlfile)
        except AttributeError:
            ymlfile.write('#BEST-FIT DRONE TIMESTAMP TIMING OFFSET NOT FOUND: REQUIRES Synchronization_Function()\n')
        ## Write a comment about and insert the file info section into the .yaml:
        ymlfile.close()
        if self.traceback==True:
            print('    --> file saved successfully')
        if self.traceback==False:
            pass
    def Main_Beam_Fitting(self,fit_param_directory='/hirax/GBO_Analysis_Outputs/main_beam_fits/',freqs=np.arange(1024),theta_solve=False,FMB_ampbound=0.999,coordbounds=[80.0,80.0,150.0],Vargs='None',t_bounds=None,copol_angle_deg=0.0):
        if self.traceback==True:
            print('Performing 2DGauss and Airy fits for [{}]chans x [{}]freqs:'.format(self.n_channels,len(freqs)))
        if self.traceback==False:
            pass
        ## chan_indices/freq_indices are captured explicitly (rather than left
        ## as the bare `range(self.n_channels)`/`freqs` passed to Fit_Main_Beam)
        ## so they can also be saved below -- this makes this function's .npz
        ## schema (chan_indices/freq_MHz/G_popt keys) match
        ## Synchronization_Function()'s saved .npz exactly, so
        ## pu.Plot_Sync_FWHM_vs_Frequency() can read either file
        ## interchangeably instead of only understanding the sync one:
        chan_indices=np.arange(self.n_channels)
        freq_indices=np.asarray(freqs)
        A_popt,A_PR,G_popt,G_PR=fu.Fit_Main_Beam(inputconcat=self,chans=chan_indices,freqs=freq_indices,coordbounds=coordbounds,theta_solve=theta_solve,ampbound=FMB_ampbound,Vargs=Vargs,t_bounds=t_bounds,copol_angle_deg=copol_angle_deg)
        self.A_popt=A_popt
        self.A_PR=A_PR
        self.G_popt=G_popt
        self.G_PR=G_PR
        ## Same beam-center convenience fields as Synchronization_Function
        ## (see its matching changelog entry/comment) -- kept under the same
        ## unprefixed naming as this function's other outputs (A_popt/G_popt
        ## etc, vs. Synchronization_Function's sync_-prefixed versions):
        self.x0=G_popt[:,:,1]
        self.y0=G_popt[:,:,3]
        self.center_xy=np.column_stack([
            np.nanmedian(self.x0,axis=1),
            np.nanmedian(self.y0,axis=1),
        ])
        if self.traceback==True:
            if self.save_traceback==True:
                print('  --> Saving output fit parameters as an .npz filetype:')
                tmpfitpath=fit_param_directory+self.Output_Prefix+'_2dGauss_and_Airy_Params.npz'
                np.savez(tmpfitpath,
                         chan_indices=chan_indices,
                         chan_prod=self.chan_prod[chan_indices],
                         freq_indices=freq_indices,
                         freq_MHz=self.freq[freq_indices],
                         A_popt=A_popt,A_PR=A_PR,G_popt=G_popt,G_PR=G_PR,
                         x0=self.x0,y0=self.y0,center_xy=self.center_xy,
                         ## Previously omitted here even though the 09/16
                         ## changelog said copol_angle_deg was recorded for
                         ## both Synchronization_Function() and
                         ## Main_Beam_Fitting() -- without it, a saved file
                         ## fit with a nonzero rotation had no record of which
                         ## frame its xsig/ysig (and therefore FWHM) were in:
                         copol_angle_deg=copol_angle_deg)
                print('  --> {}'.format(self.Output_Prefix+'_2dGauss_and_Airy_Params.npz'))
        if self.traceback==False:
            pass
    def Distance_Compensation(self,f_ind=900,plot_channels=[0],t_bounds=None):
        """
        NOT FULLY GENERALIZED YET for cross channels. drone_xyz_per_dish_interp
        is a single per-*port* distance track (one row per physical port, in
        chmap order) -- there's no ambiguity for an auto channel (it has one
        port), but a cross channel spans two ports/dishes, and there are
        multiple physically-reasonable ways to combine their two distance
        tracks into one compensation curve (average distance? compensate each
        leg separately and multiply? something else?). Rather than silently
        picking one, cross channels are currently left UNCOMPENSATED (factor
        of 1.0) here, with a printed flag. Auto channels are correctly
        resolved (port number -> chmap position -> dish index), including for
        an arbitrary/dynamic chmap subset.
        """
        if hasattr(self,'V_bgsub'):
            if hasattr(self,'drone_xyz_per_dish_interp'):
                if self.traceback==True:
                    print('Applying correction to V_bgsub using concat.Distance_Compensation() Function:')
        if t_bounds is None:
            t_bounds=[0,-1]
        ## Only use times within t_bounds when locating the closest approach
        ## (r0) used to normalize each channel's distance-compensation curve:
        search_t_index=self.t_index[t_bounds[0]:t_bounds[1]]
        r2coeffmatrix=np.NaN*np.ones((int(self.V.shape[0]),1,int(self.V.shape[2])))
        rmin=np.zeros(self.n_channels).astype(int)
        r0=np.zeros(self.n_channels)
        chan_port_idx=np.full(self.n_channels,-1,dtype=int)
        for i in range(self.n_channels):
            ## Cross channels: not yet resolved to a single distance track --
            ## see docstring above. Leave uncompensated and flag it:
            if not self.chan_is_auto[i]:
                r2coeffmatrix[:,:,i]=1.0
                if self.traceback==True:
                    pa,pb=self.chan_prod[i]
                    print('  --> Channel {} (Cross {}x{}): distance compensation NOT applied (unresolved for cross channels, see Distance_Compensation docstring).'.format(i,pa,pb))
                continue
            ## Channels with no associated dish (NaN coordinates) have nothing
            ## to compensate for -- leave their compensation factor as 1.0:
            if np.any(np.isnan(self.chan_dish_coords[i,0:2])):
                r2coeffmatrix[:,:,i]=1.0
                continue
            pa,pb=self.chan_prod[i]
            port_idx=self._port_to_dish_idx.get(int(pa))
            if port_idx is None:
                r2coeffmatrix[:,:,i]=1.0
                continue
            chan_port_idx[i]=port_idx
            rdist=np.abs(np.sqrt(((self.drone_xyz_per_dish_interp[port_idx,:,0]**2.0)+(self.drone_xyz_per_dish_interp[port_idx,:,1]**2.0)+(self.drone_xyz_per_dish_interp[port_idx,:,2]**2.0))))
            if np.all(np.isnan(rdist[search_t_index])):
                r2coeffmatrix[:,:,i]=1.0
                continue
            rmin[i]=int(search_t_index[np.nanargmin(rdist[search_t_index])])
            r0[i]=rdist[rmin[i]]
            r2coeffmatrix[:,:,i]=((rdist**2.0)/(r0[i]**2.0)).reshape((len(rdist),1))
        ## Multiply V_bgsub by the distance compensation matrix (V_bgsub is
        ## complex-native; a real-valued coefficient array keeps it complex):
        compensated_V_bgsub=r2coeffmatrix*self.V_bgsub
        ## Plot with the traceback. Each requested channel gets its own row of
        ## diagnostic plots -- no assumption is made that channels come in
        ## X/Y polarization pairs:
        if self.traceback==True:
            nplot=len(plot_channels)
            fig,axgrid=subplots(nrows=nplot,ncols=3,figsize=(18,4*nplot),squeeze=False)
            titles=['Distance_Compensation Matrix vs X','Distance_Compensation Matrix vs Y','Before/After vs X']
            for row,k in enumerate(plot_channels):
                pa,pb=self.chan_prod[k]
                if (not self.chan_is_auto[k]) or chan_port_idx[k]<0 or np.any(np.isnan(self.chan_dish_coords[k,0:2])):
                    for col in range(3):
                        axgrid[row,col].set_title('Channel {} - not compensated (cross or no dish coordinates)'.format(k))
                    continue
                pidx=chan_port_idx[k]
                ax1,ax2,ax3=axgrid[row,0],axgrid[row,1],axgrid[row,2]
                ## Plot r2coeffmatrix vs position
                ax1.plot(self.drone_xyz_per_dish_interp[pidx,:,0],r2coeffmatrix[:,0,k],'.',alpha=0.05,label='Channel {}'.format(k))
                ax1.plot(self.drone_xyz_per_dish_interp[pidx,rmin[k],0],r2coeffmatrix[rmin[k],0,k],'rx',label='minima')
                ax2.plot(self.drone_xyz_per_dish_interp[pidx,:,1],r2coeffmatrix[:,0,k],'.',alpha=0.05,label='Channel {}'.format(k))
                ax2.plot(self.drone_xyz_per_dish_interp[pidx,rmin[k],1],r2coeffmatrix[rmin[k],0,k],'rx',label='minima')
                ## Plot before/after spectra vs x
                try:
                    onmask=self.inds_on
                except AttributeError:
                    onmask=self.t_index
                ax3.plot(self.drone_xyz_per_dish_interp[pidx,:,0][onmask],np.abs(self.V_bgsub[:,f_ind,k])[onmask],'.',alpha=0.25,label='Before')
                ax3.plot(self.drone_xyz_per_dish_interp[pidx,:,0][onmask],np.abs(compensated_V_bgsub[:,f_ind,k])[onmask],'.',alpha=0.25,label='After')
                ax3.axvline(self.drone_xyz_per_dish_interp[pidx,rmin[k],0])
                for ci,ax in enumerate([ax1,ax2,ax3]):
                    ax.set_title('Channel {}: {}'.format(k,titles[ci]))
                    ax.legend(loc=1)
                ax1.set_xlabel('x')
                ax2.set_xlabel('y')
                ax3.set_xlabel('x')
                ax1.set_ylabel('$DCM[r]$')
                ax2.set_ylabel('$DCM[r]$')
                ax3.set_ylabel('$ADU^2$')
            tight_layout()
            ## Save figure?
            if self.save_traceback==True:
                print('  --> Saving output plot.')
                savefig(self.Output_Directory+self.Output_Prefix+"_Distance_Compensation_Verification.png")
            ## Redefine variable V_bgsub:
            print('  --> Complete.')
        del self.V_bgsub
        self.V_bgsub=compensated_V_bgsub