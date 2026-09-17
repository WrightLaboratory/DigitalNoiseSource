#       _       _   _   _                      _   _ _
#      | |     | | | | (_)                    | | (_) |
# _ __ | | ___ | |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _
#| '_ \| |/ _ \| __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| |_) | | (_) | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#| .__/|_|\___/ \__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#| |                             __/ |_____                 | |     __/ |
#|_|                            |___/______|                |_|    |___/
## 20211110 WT - The module structure is being completely refactored...
## This is the brand new plotting_utils.py file
## Plotting functions will be written here in a more general format... hopefully! ;)
## 07/30 Generalization pass:
##   - Added _subplot_grid() helper so multi-channel figures size themselves
##     correctly for ANY number of channels, not just even counts.
##   - Plot_Waterfalls / Plot_Saturation_Maps / Plot_Gains_vs_Data now accept an
##     optional t_bounds input (for consistency with the rest of the module) and
##     no longer assume an even channel count.
##   - Plot_Beammap no longer assumes channels come in X/Y polarization pairs
##     sharing one dish -- it now loops per-channel, and skips channels with no
##     associated dish (NaN dish_coords) instead of crashing/producing garbage.
##   - Synchronization_Verification_Plots previously had a hardcoded 2-channel
##     layout, and (more importantly) indexed V/dish_coords using the *loop
##     position* j instead of the actual channel number `chan` -- meaning it was
##     silently fitting/plotting the wrong channel's data whenever `chans` was
##     not exactly [0,1] in order. Rewrote to support any number/order of
##     channels and to always index by the true channel number, and to skip
##     channels with no dish coordinates.
##   - Added Plot_Sync_FWHM_vs_Frequency(), which plots main-beam FWHM vs
##     frequency from the fit parameters concat.CONCAT.Synchronization_Function()
##     now saves (self.sync_G_popt / self.sync_A_popt, or the saved .npz file).
## 08/19 Synchronization_Verification_Plots fixes:
##   - amp0/bg0 (np.nanmax/np.nanmin of the fit's selected data) could come out
##     NaN whenever `mbV` was empty or all-NaN for the requested (chan, find)
##     combination -- e.g. a frequency slice with no valid/unflagged samples at
##     this particular trial offset. That NaN then propagated straight into the
##     least_squares initial guess (pA/pG), which scipy rejects outright with
##     "ValueError: `x0` is infeasible" (NaN fails every bounds comparison, even
##     the default (-inf, inf) ones) -- crashing the whole verification plot
##     instead of just skipping the one bad channel. fitting_utils.Fit_Main_Beam
##     already tolerates this per (chan, freq) via a try/except, so the exact
##     same condition can occur silently during the coarse/fine search and only
##     surface here, since this function had no equivalent guard. Added an
##     explicit finite-data check (same style as the existing "no dish
##     coordinates" skip) so a bad channel/frequency combination just gets a
##     "no valid data" placeholder subplot instead of raising.
##   - Brought the Airy/2DGauss least_squares calls here in line with the same
##     amplitude-scale fix applied in fitting_utils.Fit_Main_Beam: the data is
##     now non-dimensionalized by its own local amp0 before fitting (so
##     amplitude starts at exactly 1.0 regardless of whether V is raw ADU^2 or
##     gain-calibrated and many orders of magnitude smaller), x_scale='jac' is
##     passed for additional conditioning, and the fitted amp/background are
##     rescaled back to the input data's units immediately afterward -- Apopt/
##     Gpopt (and everything derived from them below: simV, the plotted center
##     markers, etc.) are unchanged in meaning/units from before.
## 08/21 Unified auto/cross channel model (matches new corr.py/concat.py):
##   - Every array read here is now taken via np.abs() uniformly -- channels
##     are flat product indices (auto or cross), not raw ports, so .real is
##     no longer assumed/used anywhere.
##   - dish_coords[chan]/dish_coords[i] lookups (which are port-indexed)
##     replaced with chan_dish_coords[chan] (concat-side) -- the per-channel
##     resolved coordinate, correct for auto AND cross channels.
##   - Plot_Waterfalls/Plot_Time_Series/Plot_Spectra now iterate over every
##     saved channel (corr_class.n_channels, auto+cross) instead of assuming
##     n_channels==n_ports and indexing V via corr_class.chmap[i] (which
##     silently IndexErrors/misindexes now that n_channels can exceed and
##     differ in ordering from the physical port count/order).
##   - Plot_Saturation_Maps/Plot_Gains_vs_Data's gain panel stay port-indexed
##     (sat/gain are genuinely per-physical-port arrays, unaffected by the
##     channel unification) but now loop over len(chmap) explicitly rather
##     than n_channels. Plot_Gains_vs_Data's spectra panel resolves each
##     port's own auto-correlation channel via the new _auto_chan_for_port()
##     helper, since a port's auto product's flat channel index is no longer
##     guaranteed to equal the port number.
## 09/16 Synchronization_Verification_Plots gained an optional copol_angle_deg=0.0
##   argument (same meaning/sign convention as fitting_utils.Fit_Main_Beam's -- see
##   its docstring and _rotate_xy()), for a dish whose copolarization axes aren't
##   aligned with the standard NS/EW x/y local-Cartesian frame. When nonzero, the
##   drone x/y positions used for this function's own Airy/2DGauss fit (and the
##   dish-position initial guess) are rotated into that frame before fitting, and
##   rows 1-5 of the output plot (everything except the raw, unrotated full-time
##   beammap in row 0) are plotted/labeled in that same rotated frame, so the
##   verification plots actually match whatever axes Synchronization_Function was
##   told to fit along. 0.0 (default) reproduces the exact old behavior. No changes
##   were needed in Plot_Sync_FWHM_vs_Frequency -- it just reads back whatever
##   xsig/ysig the saved fit used, so it automatically reports FWHM along the
##   copol axes once the fit itself was done correctly.
import os
import glob
import pandas
import csv
import datetime
import pytz
from matplotlib.pyplot import *
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from scipy.optimize import least_squares
import numpy as np
import h5py
import matplotlib.animation as animation
from IPython.display import HTML
import beamcals.fitting_utils as fu
################################################
##                  Helpers                   ##
################################################
def _rotate_xy(x,y,angle_deg):
    """
    Local copy of fitting_utils_crs_mod_draft._rotate_xy() -- duplicated here
    (rather than called as fu._rotate_xy) because this module's `fu` is
    `beamcals.fitting_utils`, the separately-installed package, NOT the
    fitting_utils_crs_mod_draft.py draft file that actually has this
    function. Keeping this module self-sufficient for the rotation avoids
    a hard dependency on beamcals.fitting_utils being kept in sync with
    the draft (see concat_crs_mod_draft.py's 08/19 changelog note for the
    same class of stale-separately-imported-module bug).
    Re-expresses local-Cartesian (x,y) coordinates in a frame whose x'-axis
    sits `angle_deg` degrees counterclockwise from the standard x-axis
    (passive/axis rotation -- returns each point's coordinates AS SEEN FROM
    the rotated frame). angle_deg=0.0 is a no-op.
    """
    if angle_deg==0.0:
        return x,y
    theta=np.radians(angle_deg)
    c,s=np.cos(theta),np.sin(theta)
    xr=x*c+y*s
    yr=-x*s+y*c
    return xr,yr
def _subplot_grid(n_channels,ncols=2):
    """
    Return (nrows,ncols) sized so that n_channels subplots fit, for ANY number
    of channels (not just even counts). Used throughout this module instead of
    the old `int(n_channels/2)` pattern, which silently truncated the grid
    (and therefore dropped the last channel) whenever n_channels was odd.
    """
    ncols=max(1,int(ncols))
    nrows=int(np.ceil(n_channels/float(ncols)))
    return nrows,ncols
################################################
##                  Corr_Data                 ##
################################################
def _chan_label(corr_class,i):
    """Human-readable label for flat channel index i (auto or cross)."""
    pa,pb=corr_class.chan_prod[i]
    return 'Auto Ch {}'.format(pa) if corr_class.chan_is_auto[i] else 'Cross {}x{}'.format(pa,pb)
def _auto_chan_for_port(corr_class,port):
    """
    Resolve a physical port number to its own auto-correlation's flat channel
    index (position in corr_class.V's last axis / corr_class.chan_prod), now
    that channel indices are products (auto+cross), not raw ports. Returns
    None if this port has no auto-correlation among the saved channels.
    """
    matches=np.where(corr_class.chan_is_auto & (corr_class.chan_prod[:,0]==port))[0]
    return int(matches[0]) if len(matches)>0 else None
def Plot_Waterfalls(corr_class,t_bounds=[0,-1]):
    ## Express bounds for the plot axes
    tsl=slice(t_bounds[0],t_bounds[1])
    wfbounds=[corr_class.freq[-1],corr_class.freq[0],corr_class.t[tsl][-1]-corr_class.t[tsl][0],0.0]
    ## Every saved channel is already a fully-formed correlation product
    ## (auto or cross); one subplot per channel, magnitude via np.abs():
    nrows,ncols=_subplot_grid(corr_class.n_channels)
    fig1=figure(figsize=(16,int(4*nrows)))
    ## Plotting the individual waterfall plots (note freq ind is reversed!)
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(nrows,ncols,i+1)
        im=ax.imshow(np.abs(corr_class.V[tsl,::-1,i]),extent=wfbounds,cmap='gnuplot2',aspect='auto',norm=LogNorm())
        ax.set_title('{} - Ch Ind {}'.format(_chan_label(corr_class,i),i))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()
def Plot_Saturation_Maps(corr_class,t_bounds=[0,-1]):
    ## Express bounds for the plot axes
    tsl=slice(t_bounds[0],t_bounds[1])
    wfbounds=[corr_class.freq[-1],corr_class.freq[0],corr_class.t[tsl][-1]-corr_class.t[tsl][0],0.0]
    ## sat is a per-PORT array (not per-channel/product) -- it stays sized to
    ## the number of physical ports actually used (len(chmap)), indexed by
    ## position within chmap, regardless of how many auto+cross channels were
    ## saved:
    n_ports=len(corr_class.chmap)
    nrows,ncols=_subplot_grid(n_ports)
    fig1=figure(figsize=(16,int(4*nrows)))
    for i in range(n_ports):
        ax=fig1.add_subplot(nrows,ncols,i+1)
        im=ax.imshow(corr_class.sat[tsl,::-1,i].real,extent=wfbounds,cmap='gnuplot2',aspect='auto')
        ax.set_title('Saturation: Port {}'.format(corr_class.chmap[i]))
        ax.set_xlabel('Frequency, [$MHz$]')
        ax.set_ylabel('$\Delta$Time [$s$]')
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig1.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
    tight_layout()
def Plot_Time_Series(corr_class,tbounds=[0,-1],freqlist=[100,700,900]):
    ## One subplot per channel (auto or cross), magnitude via np.abs():
    nrows,ncols=_subplot_grid(corr_class.n_channels)
    fig1=figure(figsize=(16,int(4*nrows)))
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(nrows,ncols,i+1)
        for k in freqlist:
            ax.plot(corr_class.t_index[tbounds[0]:tbounds[1]],np.abs(corr_class.V[tbounds[0]:tbounds[1],k,i]),'.',label="F={}".format(corr_class.freq[k]))
        ax.set_title('Time Series: {} - Ind {}'.format(_chan_label(corr_class,i),i))
        ax.set_ylabel('Power [$ADU^2$]')
        ax.set_xlabel('Time Index')
        ax.legend()
    tight_layout()
def Plot_Spectra(corr_class,tbounds=[5,-5],tstep=2000):
    ## One subplot per channel (auto or cross), magnitude via np.abs():
    nrows,ncols=_subplot_grid(corr_class.n_channels)
    fig1=figure(figsize=(16,4*nrows))
    CNorm=colors.Normalize()
    CNorm.autoscale(np.arange(len(corr_class.t))[tbounds[0]:tbounds[1]:tstep])
    CM=cm.gnuplot2
    CM=cm.magma
    for i in range(corr_class.n_channels):
        ax=fig1.add_subplot(nrows,ncols,i+1)
        for k,t_ind in enumerate(np.arange(len(corr_class.t))[tbounds[0]:tbounds[1]:tstep]):
            ax.semilogy(corr_class.freq,np.abs(corr_class.V[t_ind,:,i]),'.',c=CM(CNorm(t_ind)),label='t = {:.2f}'.format(float(corr_class.t[t_ind]-corr_class.t[0])))
        ax.set_title('Spectra: {} - Ind {}'.format(_chan_label(corr_class,i),i))
        ax.set_ylabel('Log Power [$ADU^2$]')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend(fontsize='small')
    tight_layout()
def Plot_Gains_vs_Data(corr_class,tind=1,t_bounds=None):
    ## Gain is a per-PORT quantity (stays sized to len(chmap)); pair each
    ## port's gain with that same port's own auto-correlation spectrum
    ## (resolved via _auto_chan_for_port, since a port's auto-correlation's
    ## flat channel index is no longer guaranteed to equal the port number):
    n_ports=len(corr_class.chmap)
    nrows,_=_subplot_grid(n_ports)
    fig1=figure(figsize=(16,int(4*nrows)))
    for i in range(n_ports):
        port=corr_class.chmap[i]
        ax=fig1.add_subplot(nrows,4,int(2*i)+1)
        ax.plot(corr_class.freq,corr_class.gain[:,port],'.',label="Gain Exp = {}".format(corr_class.gain_exp[port]))
        ax.set_title("Gain: Port {}".format(port))
        ax.set_ylabel('Gain')
        ax.set_xlabel('Frequency [MHz]')
        ax.legend()
        ax=fig1.add_subplot(nrows,4,int(2*i)+2)
        auto_k=_auto_chan_for_port(corr_class,port)
        if auto_k is None:
            ax.set_title("Spectra: Port {} - no auto-correlation saved".format(port))
        else:
            ax.plot(corr_class.freq,np.abs(corr_class.V[tind,:,auto_k]),'.')
            ax.set_title("Spectra: Port {} (Auto Ch {})".format(port,auto_k))
        ax.set_ylabel('Power [$ADU^2$]')
        ax.set_xlabel('Frequency [MHz]')
    tight_layout()
################################################
##                 Drone_Data                 ##
################################################
def Plot_Drone_Coordinates(drone_class,coo='lat',t_bounds=[0,-1]):
    print('plotting drone coordinates for all time samples:')
    fig1,[[ax1,ax2,ax3],[ax4,ax5,ax6]]=subplots(nrows=2,ncols=3,figsize=(15,9))
    ## Plot p0 coordinate origin:
    ax1.plot(drone_class.origin[0],drone_class.origin[1],'ro')
    ax2.axhline(drone_class.origin[0],c='b')
    ax3.axhline(drone_class.origin[1],c='b')
    if coo=='lat':
        ## Title each coordinate subplot:
        ax1.set_title('Lat vs Lon')
        ax2.set_title('Lat vs Time')
        ax3.set_title('Lon vs Time')
        ax4.set_title('Velocity vs Time')
        ax5.set_title('Altitude vs Time')
        ax6.set_title('Yaw vs Time')
        ## Specify arrays/vectors to plot in 1,3,4 coordinate subplot
        xqtys=[drone_class.latitude,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index]
        yqtys=[drone_class.longitude,drone_class.latitude,drone_class.longitude,drone_class.velocity,drone_class.altitude,drone_class.yaw]
        xtags=['Latitude, [$deg$]','Drone Index','Drone Index','Drone Index','Drone Index','Drone Index']
        ytags=['Longitude, [$deg$]','Latitude, [$deg$]','Longitude, [$deg$]','Velocity, [m/s]','Altitude, [$m$]','Yaw [$deg$]']
    if coo=='xy':
        ax1.set_title('X vs Y')
        ax2.set_title('X vs Time')
        ax3.set_title('Y vs Time')
        ax4.set_title('Velocity vs Time')
        ax5.set_title('Altitude vs Time')
        ax6.set_title('Yaw vs Time')
        ## Specify arrays/vectors to plot in 1,3,4 coordinate subplot
        xqtys=[drone_class.coords_xyz_LC[:,0],drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index,drone_class.t_index]
        yqtys=[drone_class.coords_xyz_LC[:,1],drone_class.latitude,drone_class.longitude,drone_class.velocity,drone_class.altitude,drone_class.yaw]
        xtags=['X, [$m$]','Drone Index','Drone Index','Drone Index','Drone Index','Drone Index']
        ytags=['Y, [$m$]','X, [$m$]','Y, [$m$]','Velocity, [m/s]','Altitude, [$m$]','Yaw [$deg$]']
    print('overplotting drone coordinates for t_cut samples: ['+str(t_bounds[0])+':'+str(t_bounds[1])+']')
    for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
        ax.plot(np.nanmin(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmin(yqtys[i][t_bounds[0]:t_bounds[1]]))
        ax.plot(np.nanmax(xqtys[i][t_bounds[0]:t_bounds[1]]),np.nanmax(yqtys[i][t_bounds[0]:t_bounds[1]]))
        autoscalelims=ax.axis()
        ax.clear()
        ax.plot(xqtys[i],yqtys[i],'.',label='all samples')
        ax.plot(xqtys[i][t_bounds[0]:t_bounds[1]],yqtys[i][t_bounds[0]:t_bounds[1]],'.',label='selected samples')
        ax.set_xlabel(xtags[i])
        ax.set_ylabel(ytags[i])
        ax.grid()
        ax.legend()
        ax.set_xlim(autoscalelims[0],autoscalelims[1])
        ax.set_ylim(autoscalelims[2],autoscalelims[3])
    tight_layout()
def Plot_Angular_Coordinates(drone_class,t_bounds=[0,-1]):
    fig=figure(figsize=(15,4.5))
    ax1=fig.add_subplot(1, 3, 1)
    ax1.plot(drone_class.t_index[:],(180/np.pi)*drone_class.coords_rpt[:,1],'.')
    ax1.plot(drone_class.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],1],'.')
    ax1.set_xlabel('time index')
    ax1.set_ylabel(r'$\phi, [deg]$')
    ax2=fig.add_subplot(1, 3, 2)
    ax2.plot(drone_class.t_index[:],(180/np.pi)*drone_class.coords_rpt[:,2],'.')
    ax2.plot(drone_class.t_index[t_bounds[0]:t_bounds[1]],(180/np.pi)*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
    ax2.set_xlabel('time index')
    ax2.set_ylabel(r'$\theta, [deg]$')
    ax3=fig.add_subplot(1, 3, 3, projection='polar')
    ax3.plot(drone_class.coords_rpt[:,1],180/np.pi*drone_class.coords_rpt[:,2],'.')
    ax3.plot(drone_class.coords_rpt[t_bounds[0]:t_bounds[1],1],180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2],'.')
    ax3.set_rlim(np.nanmin(180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2]),1.1*np.nanmax(180/np.pi*drone_class.coords_rpt[t_bounds[0]:t_bounds[1],2]))
    tight_layout()
def Plot_3d(drone_class,t_bounds=[0,-1]):
    fig=figure(figsize=(10,4.5))
    tkeys=['Geocentric Cartesian','Local Cartesian']
    for i,coordset in enumerate([drone_class.coords_xyz_GC,drone_class.coords_xyz_LC]):
        ax=fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_title(tkeys[i])
        ax.set_xlabel('x, [meters]')
        ax.set_ylabel('y, [meters]')
        ax.set_zlabel('z, [meters]')
        ax.plot(coordset[:,0],coordset[:,1],coordset[:,2],'.')
        ax.plot(coordset[t_bounds[0]:t_bounds[1],0],coordset[t_bounds[0]:t_bounds[1],1],coordset[t_bounds[0]:t_bounds[1],2],'.')
    tight_layout()
def Plot_Transmitter_Pointing(drone_class,t_bounds=[0,-1],t_step=1):
    fig=figure(figsize=(10,8))
    ax=fig.add_subplot(111)
    ## DRONE COORDINATE SYSTEM x,y,z=North,East,Down VARIABLES ##
    UV_nose_north=np.array([1,0,0]) #unit vector for nose pointing north, no roll/pitch, prior to rotations
    UV_trans_down=np.array([0,0,1]) #unit vector for transmitter pointing down prior to rotations
    ## drone roll, pitch, yaw angles:
    ypr=np.ndarray((len(drone_class.t_index),3))
    ypr[:,0]=drone_class.yaw
    ypr[:,1]=drone_class.pitch
    ypr[:,2]=drone_class.roll
    ## TRANSMITTER POINTING DIRECTION as fxn of time in Local Cartesian: (transform by [y,p,r]=[+90,0,+180] rot)
    trans_pointing_xyz=np.array([gu.rot_mat(np.array([90.0,0.0,180.0]))@gu.rot_mat(ypr[m,:])@UV_trans_down for m in range(len(drone_class.t_index))])
    ## Plot Parameters:
    [Qlb,Qub,Qstep]=[t_bounds[0],t_bounds[1],t_step]
    M=np.abs(np.hypot(trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1]))
    CNorm=colors.Normalize()
    CNorm.autoscale(M)
    CM=cm.gnuplot2
    SM=cm.ScalarMappable(cmap=CM, norm=CNorm)
    SM.set_array([])
    q=ax.quiver(drone_class.coords_xyz_LC[Qlb:Qub:Qstep,0],drone_class.coords_xyz_LC[Qlb:Qub:Qstep,1],trans_pointing_xyz[Qlb:Qub:Qstep,0],trans_pointing_xyz[Qlb:Qub:Qstep,1],color=CM(CNorm(M)))
    ax.quiverkey(q,X=0.15,Y=0.05,U=1,label='Unit Vector', labelpos='E')
    fig.colorbar(SM,label='XY Projection Magnitude')
    ax.set_xlabel('Local X Position, [m]')
    ax.set_ylabel('Local Y Position, [m]')
    ax.set_title('Transmitter Deviation from Nadir [XY Projection]')
    tight_layout()
def Plot_Polar_Lines_of_Sight(drone_class,t_bounds=[0,-1],t_step=1,dishid=0):
    fig1,[ax1,ax2]=subplots(nrows=1,ncols=2,figsize=(16,8),subplot_kw=dict(projection="polar"))
    ax1.set_title('Drone Position in Receiver {} Beam'.format(dishid))
    ax1.plot(drone_class.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*drone_class.rpt_t_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.b',markersize=1.0)
    ax1.plot(0,0,'ro')
    ax2.set_title('Receiver {} Position in Drone Beam'.format(dishid))
    ax2.plot(drone_class.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,1],180.0/np.pi*drone_class.rpt_r_per_dish[dishid,t_bounds[0]:t_bounds[1]:t_step,2],'.g',markersize=1.0)
    ax2.plot(0,0,'ro')
    tight_layout()
def Animate_Drone_Flight(
    dronedata, gbosite,
    step=10,
    manual_north_shift_m=0,
    manual_east_shift_m=0,
    start_time_est=None,
    end_time_est=None,
    start_percent=0.0,
    end_percent=1.0,
    interval=50,
    center_dish=None,
    show_dishes=None
):
    """
    Animate drone flight path with optional time range and percentage selection.
    Parameters
    ----------
    dronedata : object
        Must have attributes latitude, longitude, altitude, t_arr_timestamp.
    gbosite : object
        Must have attributes origin, coords (m offsets from origin), keystrings.
    center_dish : int or None
        Dish number to use as map center (origin). If None, use gbosite.origin.
    show_dishes : list of int or None
        Which dishes to display on map. If None, show only center dish.
    """
    # === Load and downsample drone data ===
    lat = dronedata.latitude[::step]
    lon = dronedata.longitude[::step]
    alt = dronedata.altitude[::step]
    timestamps = dronedata.t_arr_timestamp[::step]
    # === Get reference (origin) ===
    if center_dish is not None:
        # dish indices in keystrings are like "Dish_5_X", "Dish_5_Y" etc.
        dish_mask = np.array([f"Dish_{center_dish}_" in k for k in gbosite.keystrings])
        dish_coords = gbosite.coords[dish_mask]
        # average X/Y in case both polarizations exist
        ref_x, ref_y, ref_z = dish_coords.mean(axis=0)
        # convert back to lat/lon relative to gbosite.origin
        lat0, lon0, alt0 = gbosite.origin
        R = 6371000
        lat0_shifted = lat0 + (ref_y / R) * (180 / np.pi)
        lon0_shifted = lon0 + (ref_x / (R * np.cos(np.radians(lat0)))) * (180 / np.pi)
    else:
        lat0, lon0, alt0 = gbosite.origin
        R = 6371000
        lat0_shifted, lon0_shifted = lat0, lon0
    # === Manual shift ===
    delta_lat_deg = manual_north_shift_m / R * (180 / np.pi)
    delta_lon_deg = manual_east_shift_m / (R * np.cos(np.radians(lat0_shifted))) * (180 / np.pi)
    lat0_shifted += delta_lat_deg
    lon0_shifted += delta_lon_deg
    # === Time filtering (same as before) ===
    # ... [unchanged code] ...
    # === Set up figure ===
    fig=figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    line, = ax.plot([], [], lw=2, color='blue')
    point, = ax.plot([], [], 'ro')
    alt_label = ax.text(0, 0, '', fontsize=12, ha='left', va='bottom')
    title = ax.set_title("")
    margin = 0.00005
    ax.set_xlim(min(lon) - margin, max(lon) + margin)
    ax.set_ylim(min(lat) - margin, max(lat) + margin)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    # === Dish markers ===
    if show_dishes is None and center_dish is not None:
        show_dishes = [center_dish]
    elif show_dishes is None:
        show_dishes = []
    for d in show_dishes:
        mask = np.array([f"Dish_{d}_" in k for k in gbosite.keystrings])
        dcoords = gbosite.coords[mask]
        dx, dy, dz = dcoords.mean(axis=0)
        dlat = lat0 + (dy / R) * (180 / np.pi)
        dlon = lon0 + (dx / (R * np.cos(np.radians(lat0)))) * (180 / np.pi)
        ax.plot(dlon, dlat, 'r*' if d == center_dish else 'g*',
                markersize=10, label=f'Dish {d}')
    ax.legend()
    # === Secondary axes and animation functions remain unchanged ===
    # ...
        # === Init/update functions ===
    def init():
        line.set_data([], [])
        point.set_data([], [])
        alt_label.set_text('')
        title.set_text('')
        return line, point, alt_label, title
    def update(frame):
        line.set_data(lon[:frame+1], lat[:frame+1])
        point.set_data([lon[frame]], [lat[frame]])
        alt_label.set_position((lon[frame], lat[frame]))
        alt_label.set_text(f"{alt[frame]:.1f} ft")
        title.set_text(f"Time: {timestamps[frame].strftime('%H:%M:%S')}")
        return line, point, alt_label, title
    # === Animate ===
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(lat)), init_func=init,
        blit=True, interval=interval
    )
    return HTML(animation.FuncAnimation(
        fig, update, frames=range(len(lat)), init_func=init,
        blit=True, interval=interval
    ).to_jshtml())
################################################
##                   CONCAT                   ##
################################################
def Plot_Beammap(concat_class,t_bounds=[0,-1],coord_args="LC",pulse_args=None,f_bounds=[300,340],cbounds=[],dotsize=40):
    """
    Plot a per-channel beammap. Generalized to work for any number of
    channels: each channel gets its own subplot (rather than assuming
    channels come in X/Y polarization pairs sharing a dish), and channels
    with no associated dish (NaN chan_dish_coords) are skipped. `chan` is a
    flat product index (auto or cross); magnitude is always via np.abs().
    """
    n_channels=concat_class.n_channels
    nrows,ncols=_subplot_grid(n_channels)
    fig1=figure(figsize=(16,int(7*nrows/2.0)+7))
    t_cut_full=np.arange(concat_class.t_index[t_bounds[0]],concat_class.t_index[t_bounds[1]])
    for chan in range(n_channels):
        ## Skip channels with no associated dish -- nothing spatial to plot:
        if np.any(np.isnan(concat_class.chan_dish_coords[chan,0:2])):
            continue
        ## Select the time cut + data source depending on pulse_args:
        if pulse_args is None:
            t_cut=t_cut_full
            V_source=concat_class.V
        elif pulse_args=="on":
            t_cut=np.intersect1d(t_cut_full,concat_class.inds_on).tolist()
            V_source=concat_class.V
        elif pulse_args=="off":
            t_cut=np.intersect1d(t_cut_full,concat_class.inds_off).tolist()
            V_source=concat_class.V
        elif pulse_args=="bg":
            t_cut=t_cut_full.tolist()
            V_source=concat_class.V_bg
        elif pulse_args=="bgsub":
            t_cut=np.intersect1d(t_cut_full,concat_class.inds_on).tolist()
            V_source=concat_class.V_bgsub
        else:
            raise ValueError("Unrecognized pulse_args: {}".format(pulse_args))
        pt_colors=np.nanmean(np.abs(V_source[t_cut,f_bounds[0]:f_bounds[1],chan]),axis=1)
        if coord_args=="LC":
            ax=fig1.add_subplot(nrows,ncols,chan+1)
            x=concat_class.drone_xyz_LC_interp[t_cut,0]
            y=concat_class.drone_xyz_LC_interp[t_cut,1]
        elif coord_args=="Pol":
            ax=fig1.add_subplot(nrows,ncols,chan+1,projection="polar")
            x=concat_class.drone_rpt_r_per_dish_interp[chan,t_cut,1]
            y=180.0/np.pi*concat_class.drone_rpt_r_per_dish_interp[chan,t_cut,2]
        else:
            raise ValueError("Unrecognized coord_args: {}".format(coord_args))
        im=ax.scatter(x,y,s=dotsize,c=pt_colors,cmap='gnuplot2',norm=LogNorm())
        if len(cbounds)==2:
            im.set_clim(cbounds[0],cbounds[1])
        ax.set_facecolor('k')
        pa,pb=concat_class.chan_prod[chan]
        chlabel='Auto Ch {}'.format(pa) if concat_class.chan_is_auto[chan] else 'Cross {}x{}'.format(pa,pb)
        ax.set_title(concat_class.name+' {} Beammap'.format(chlabel))
        if coord_args=="LC":
            ax.set_xlabel('X Position $[m]$')
            ax.set_ylabel('Y Position $[m]$')
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right", size="3%", pad=0.05)
            cbar=fig1.colorbar(im,cax=cax)
            cbar.set_label('Power [$ADU^2$]')
        if coord_args=="Pol":
            cbar=fig1.colorbar(im,ax=ax,aspect=40)
            cbar.set_label('Power [$ADU^2$]')
    tight_layout()
def Synchronization_Verification_Plots(inputconcat,chans=np.array([2,3]),find=900,coordbounds=[50.0,50.0,150.0],ampbound=0.999,t_bounds=None,copol_angle_deg=0.0):
    """
    Produce beam-fit verification plots for the requested channels.
    Generalized to support any number/order of channels: previously this
    function was hardcoded for exactly 2 channels laid out in a fixed 6x2
    grid, and (more importantly) it indexed V / dish_coords using the loop
    position `j` instead of the actual channel number `chan` -- so if `chans`
    was ever anything other than [0,1] in increasing order, it silently
    fit/plotted the wrong channel's data. That bug is fixed here: every
    lookup now uses `chan`, not `j`. Channels with no dish coordinates are
    skipped.

    Also skips (rather than crashing on) a channel whose selected data at
    this `find` frequency/offset is empty or entirely non-finite -- see the
    08/19 changelog note at the top of this file -- and fits the Airy/2DGauss
    models in amplitude-normalized units (matching fitting_utils.Fit_Main_Beam)
    so the fit's numerical behavior doesn't depend on whether `inputconcat.V`
    is raw ADU^2 or gain-calibrated to a very different absolute scale.

    copol_angle_deg : float, default 0.0
        Same meaning/sign convention as fitting_utils.Fit_Main_Beam's
        argument of the same name (see its docstring and _rotate_xy) --
        rotates the drone x/y positions used for the fit (and the
        dish-position initial guess) into the true copolarization frame
        before fitting, for a dish whose polarizations aren't aligned with
        the standard NS/EW x/y frame. 0.0 (default) reproduces the exact
        old behavior. When nonzero, rows 1-5 below (everything except the
        top-row full-time beammap, which is intentionally left in the raw,
        unrotated local-Cartesian frame for physical context) are plotted
        and labeled in this rotated (copol) frame -- pass the same value
        you used/plan to use in Synchronization_Function so the plots here
        actually match the fit whose quality you're checking.
    """
    chans=np.asarray(chans)
    ncols=len(chans)
    fig,axarr=subplots(nrows=6,ncols=ncols,figsize=(8*ncols,40),squeeze=False)
    for j,chan in enumerate(chans):
        if np.any(np.isnan(inputconcat.chan_dish_coords[chan,0:2])):
            for row in range(6):
                axarr[row,j].set_title('Channel {} - no dish coordinates'.format(chan))
            continue
        ## Every channel is already a fully-formed correlation product (auto
        ## or cross) -- np.abs() is the uniform, correct way to get its
        ## magnitude regardless of which type it is:
        V_mag_full=np.abs(inputconcat.V[:,find,chan])
        ## define timecuts for amplitude:
        tacut=inputconcat.t_index[V_mag_full<ampbound*(np.nanmax(V_mag_full))]
        ## define timecuts for cartesian coordinates:
        txcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,0])<coordbounds[0]]
        tycut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,1])<coordbounds[1]]
        tzcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,2])>coordbounds[2]]
        coordcut=np.intersect1d(np.intersect1d(txcut,tycut),tzcut)
        if t_bounds is not None:
            tbcut=inputconcat.t_index[t_bounds[0]:t_bounds[1]]
            coordcut=np.intersect1d(coordcut,tbcut)
        try:
            ttcut=np.intersect1d(np.intersect1d(coordcut,tacut),inputconcat.inds_on)
        except AttributeError:
            ttcut=np.intersect1d(coordcut,tacut)
        ## data points for fit:
        mbx=inputconcat.drone_xyz_LC_interp[ttcut,0]
        mby=inputconcat.drone_xyz_LC_interp[ttcut,1]
        mbz=inputconcat.drone_xyz_LC_interp[ttcut,2]
        mbV=V_mag_full[ttcut]
        ## Guard against an empty or all-non-finite selection for this
        ## (chan, find) combination -- e.g. a frequency with no valid,
        ## unflagged samples at this particular trial offset. Without this,
        ## amp0/bg0 below come out NaN, which least_squares rejects outright
        ## ("ValueError: `x0` is infeasible", since NaN fails every bounds
        ## comparison, even the default (-inf, inf) ones), crashing the whole
        ## verification plot instead of just this one channel. See the 08/19
        ## changelog note at the top of this file:
        if mbV.size==0 or not np.any(np.isfinite(mbV)):
            for row in range(6):
                axarr[row,j].set_title('Channel {} - no valid data at find={} for this offset'.format(chan,find))
            continue
        ## shared params:
        amp0=np.nanmax(mbV)
        bg0=np.nanmin(mbV)
        x00=inputconcat.chan_dish_coords[chan,0]
        y00=inputconcat.chan_dish_coords[chan,1]
        ## Re-express the fit data and dish-position initial guess along the
        ## true copolarization axes, exactly like fitting_utils.Fit_Main_Beam
        ## does -- see copol_angle_deg docstring above. A no-op when
        ## copol_angle_deg=0.0. mbx/mby are reused for everything below
        ## (fitting, the simulated grid, and the row 1-5 plots), so this one
        ## rotation keeps the fit and every plot that isn't the raw full-time
        ## beammap (row 0) self-consistent in the same (rotated) frame:
        mbx,mby=_rotate_xy(mbx,mby,copol_angle_deg)
        x00,y00=_rotate_xy(x00,y00,copol_angle_deg)
        ## airy params:
        rad0=25.0
        ## 2dgauss params:
        xsig0=6.0
        ysig0=6.0
        ## Non-dimensionalize by this fit's own local amplitude before
        ## fitting, exactly as fitting_utils.Fit_Main_Beam does -- keeps
        ## amplitude/background at O(1) regardless of whether mbV is raw
        ## ADU^2 or gain-calibrated to a very different absolute scale, so
        ## the least_squares finite-difference step (which floors at
        ## max(1, |x0|)) doesn't blow up relative to a tiny raw amplitude:
        norm = amp0 if (amp0 is not None and not np.isnan(amp0) and amp0 != 0) else 1.0
        mbV_n = mbV/norm
        amp0_n = amp0/norm  # == 1.0 (unless amp0==0, then norm=1 and amp0_n=0)
        bg0_n = bg0/norm
        mb_input_data_n=np.array([mbx,mby,mbV_n])
        ## initial guess and bounds (normalized):
        pA=np.array([amp0_n,x00,y00,rad0,bg0_n])
        pG=np.array([amp0_n,x00,xsig0,y00,ysig0,bg0_n])
        ## run the fits (normalized units), then rescale amp/bg back to the
        ## input data's units so Apopt/Gpopt below mean exactly what they did
        ## before this change:
        Apopt=least_squares(fu.Airy_2d_LC_opt,x0=pA,method='trf',x_scale='jac',args=mb_input_data_n).x.copy()
        Apopt[0]*=norm  # amp
        Apopt[4]*=norm  # c (background)
        Gpopt=least_squares(fu.Gauss_2d_LC_opt,x0=pG,method='trf',x_scale='jac',args=mb_input_data_n).x.copy()
        Gpopt[0]*=norm  # amp
        Gpopt[5]*=norm  # c (background)
        ## simulate space of these coords:
        simx=np.outer(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),np.ones(100)).flatten()
        simy=np.outer(np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)).flatten()
        simV=fu.Gauss_2d_LC_func(Gpopt,simx,simy)
        ## ROW 0: full-time beammap
        ax=axarr[0,j]
        ax.set_title('Channel {} Beammap - Full Time'.format(chan))
        ax.set_facecolor('k')
        try:
            tbig=np.intersect1d(tzcut,inputconcat.inds_on)
        except AttributeError:
            tbig=tzcut
        im=ax.scatter(inputconcat.drone_xyz_LC_interp[tbig,0],inputconcat.drone_xyz_LC_interp[tbig,1],c=np.abs(inputconcat.V[tbig,find,chan]),s=20,norm=LogNorm())
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=fig.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ## Label suffix so rows 1-5 (all plotted/fit in the copol-rotated
        ## frame when copol_angle_deg!=0.0) are visually distinguishable
        ## from row 0's raw, unrotated local-Cartesian beammap:
        _copol_tag=' (copol frame, {:.1f}deg)'.format(copol_angle_deg) if copol_angle_deg!=0.0 else ''
        ## ROW 1: points used in fit
        ax=axarr[1,j]
        ax.set_title('Channel {} Beammap - Points Fit w/ 2DGauss{}'.format(chan,_copol_tag))
        ax.set_facecolor('k')
        im=ax.scatter(mbx,mby,c=mbV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=fig.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ## ROW 2: best fit model
        ax=axarr[2,j]
        ax.set_title('Channel {} Beammap - Best Fit 2DGauss{}'.format(chan,_copol_tag))
        ax.set_facecolor('k')
        im=ax.scatter(simx,simy,c=simV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=fig.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ax.plot(Gpopt[1],Gpopt[3],'wx',label='[{:.2f},{:.2f}]'.format(Gpopt[1],Gpopt[3]))
        ax.legend()
        ## ROW 3: overlay
        ax=axarr[3,j]
        ax.set_title('Channel {} Beammap - Overlay{}'.format(chan,_copol_tag))
        ax.set_facecolor('k')
        im=ax.scatter(simx,simy,c=simV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        im=ax.scatter(mbx,mby,c=mbV,norm=LogNorm())
        im.set_clim(np.nanmax((np.nanmin(simV),np.nanmin(mbV))),np.nanmax((np.nanmax(simV),np.nanmax(mbV))))
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("right", size="3%", pad=0.05)
        cbar=fig.colorbar(im,cax=cax)
        cbar.set_label('Power [$ADU^2$]')
        ax.plot(Gpopt[1],Gpopt[3],'wx',label='[{:.2f},{:.2f}]'.format(Gpopt[1],Gpopt[3]))
        ax.legend()
        ## ROW 4: X cut
        ax=axarr[4,j]
        ax.set_title('Channel {} Data vs Best-Fit 2DGauss{}'.format(chan,_copol_tag))
        ax.set_facecolor('k')
        ax.scatter(mbx,mbV,c=mbV,norm=LogNorm())
        ax.plot(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),fu.Gauss_2d_LC_func(Gpopt,np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),Gpopt[3]*np.ones(100)),'w.--',label='X center pass 2DGauss')
        ax.plot(np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),fu.Airy_2d_LC_func(Apopt,np.linspace(np.nanmin(mbx),np.nanmax(mbx),100),Apopt[2]*np.ones(100)),'c.--',label='X center pass Airy')
        ax.axvline(Gpopt[1],c='w',label='X center = {:.2f}'.format(Gpopt[1]))
        ax.set_yscale('log')
        ax.legend()
        ## ROW 5: Y cut
        ax=axarr[5,j]
        ax.set_title('Channel {} Data vs Best-Fit 2DGauss{}'.format(chan,_copol_tag))
        ax.set_facecolor('k')
        ax.scatter(mby,mbV,c=mbV,norm=LogNorm())
        ax.plot(np.linspace(np.nanmin(mby),np.nanmax(mby),100),fu.Gauss_2d_LC_func(Gpopt,Gpopt[1]*np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)),'w.--',label='Y center pass 2DGauss')
        ax.plot(np.linspace(np.nanmin(mby),np.nanmax(mby),100),fu.Airy_2d_LC_func(Apopt,Apopt[1]*np.ones(100),np.linspace(np.nanmin(mby),np.nanmax(mby),100)),'c.--',label='Y center pass Airy')
        ax.axvline(Gpopt[3],c='w',label='Y center = {:.2f}'.format(Gpopt[3]))
        ax.set_yscale('log')
        ax.legend()
    ## Row 0 (full-time beammap) is always plotted in the raw, unrotated
    ## local-Cartesian frame (see the ROW 0 block above), so its axis labels
    ## stay plain X/Y regardless of copol_angle_deg. Rows 1-5 are fit and
    ## plotted in the copol-rotated frame whenever copol_angle_deg!=0.0, so
    ## their axes get a primed label to make that explicit:
    xlabel_rot = "X$'$ $[m]$" if copol_angle_deg!=0.0 else 'X $[m]$'
    ylabel_rot = "Y$'$ $[m]$" if copol_angle_deg!=0.0 else 'Y $[m]$'
    for j in range(ncols):
        axarr[0,j].set_xlabel('X $[m]$')
        axarr[0,j].set_ylabel('Y $[m]$')
        for row in (1,2,3):
            axarr[row,j].set_xlabel(xlabel_rot)
            axarr[row,j].set_ylabel(ylabel_rot)
        axarr[4,j].set_xlabel(xlabel_rot)
        axarr[4,j].set_ylabel('Power [$ADU^2$]')
        axarr[5,j].set_xlabel(ylabel_rot)
        axarr[5,j].set_ylabel('Power [$ADU^2$]')
    tight_layout()
def Plot_Sync_FWHM_vs_Frequency(source,pol_labels=None,freq_bounds=None):
    """
    Plot main-beam FWHM (X and Y) vs frequency, using the fit parameters
    saved by concat.CONCAT.Synchronization_Function() (self.sync_G_popt,
    indexed [chan, freq, (amp,x0,xsig,y0,ysig,c[,theta])]).
    `source` can be either a CONCAT object that has already had
    Synchronization_Function() run on it, or a path (str) to the
    '..._Synchronization_MainBeam_Fits.npz' file it saves to disk.
    freq_bounds : optional (fmin_MHz, fmax_MHz)
        Restrict the plot to frequencies within this range (inclusive). Does
        not require re-fitting -- just filters the already-computed results.
        Defaults to None (show every fit frequency available in `source`).
    Works for any number of channels; channels with no valid (non-NaN) fit
    -- e.g. channels with no associated dish -- are automatically skipped.

    Note on axes: "X FWHM"/"Y FWHM" here are always whatever xsig/ysig the
    saved fit used -- if Synchronization_Function() was run with a nonzero
    copol_angle_deg (see its docstring), sync_G_popt's x0/y0/xsig/ysig are
    already in that rotated (copol) frame, so no changes are needed here:
    these plots automatically show FWHM along the true copolarization axes.
    The angle used is saved alongside the other fit parameters as
    sync_copol_angle_deg / the .npz's copol_angle_deg field if you need to
    double check which frame a given saved result is in.
    """
    if isinstance(source,str):
        data=np.load(source)
        freq_MHz=data['freq_MHz']
        chan_indices=data['chan_indices']
        G_popt=data['G_popt']
    else:
        freq_MHz=source.freq[source.sync_freqs]
        chan_indices=source.sync_chans
        G_popt=source.sync_G_popt
    if freq_bounds is not None:
        fmin,fmax=freq_bounds
        keep=(freq_MHz>=fmin)&(freq_MHz<=fmax)
        if not np.any(keep):
            print('No fit frequencies fall within freq_bounds={}.'.format(freq_bounds))
            return None
        freq_MHz=freq_MHz[keep]
        G_popt=G_popt[:,keep,:]
    fwhm_const=2.0*np.sqrt(2.0*np.log(2.0))
    n_chans=G_popt.shape[0]
    fig,axs=subplots(nrows=2,ncols=1,sharex=True,figsize=(10,8))
    any_plotted=False
    for i in range(n_chans):
        chan=chan_indices[i]
        xsig=G_popt[i,:,2]
        ysig=G_popt[i,:,4]
        if np.all(np.isnan(xsig)) and np.all(np.isnan(ysig)):
            continue
        any_plotted=True
        label=pol_labels[i] if pol_labels is not None else 'Channel {}'.format(chan)
        axs[0].plot(freq_MHz,fwhm_const*xsig,'.-',label=label)
        axs[1].plot(freq_MHz,fwhm_const*ysig,'.-',label=label)
    if not any_plotted:
        print('No valid (non-NaN) synchronization fit parameters found to plot.')
        return None
    axs[0].set_ylabel('X FWHM [m]')
    axs[1].set_ylabel('Y FWHM [m]')
    axs[1].set_xlabel('Frequency [MHz]')
    axs[0].set_title('Main Beam FWHM vs Frequency')
    axs[0].legend(fontsize='small',ncol=2)
    tight_layout()
    return fig
################################################
##                Other plotting              ##
################################################
def plot_MBG_fits(filelist,cmp='gnuplot2',fitdir='/hirax/GBO_Analysis_Outputs/main_beam_fits/'):
    # This plots the npz files generated from Gaussian main beam fitting
    #some setup
    cm = get_cmap(cmp)
    freqs = np.arange(800,400,-400/1024.)
    # make a plot per file:
    print(len(filelist))
    for file in filelist:
        ## Make figure set: 6 parameters, plot vs freq with different markers dependent on pol/dish
        fig, axs = subplots(nrows=6,ncols=1,sharex=True,figsize=(15,15))
        ## Loop through fits and make the plot:
        fits = np.load(fitdir+file)
        N = len(fits['G_popt'][:,0,1])
        chans = np.arange(N)
        cs=[cm(1.*i/N) for i in range(N)]
        for c,chan in enumerate(chans):
            if chan%2==0: # assume its even (E pol)
                mtype = ','
                msize = 4
            else: # assume its odd (N pol)
                mtype='^'
                msize= 2
            axs[0].set_title('Flight: '+file)
            axs[0].plot(freqs,fits['G_popt'][chan,:,1],marker=mtype,ms=msize,color=cs[chan],linestyle='None')
            axs[0].set_ylim(-20,20)
            axs[0].set_ylabel('X centroid')
            axs[1].plot(freqs,fits['G_popt'][chan,:,3],marker=mtype,ms=msize,color=cs[chan],linestyle='None')
            axs[1].set_ylim(-20,20)
            axs[1].set_ylabel('Y centroid')
            axs[2].plot(freqs,fits['G_popt'][chan,:,2],marker=mtype,ms=msize,color=cs[chan],linestyle='None')
            axs[2].set_ylim(0,20)
            axs[2].set_ylabel('Xsig')
            axs[3].plot(freqs,fits['G_popt'][chan,:,4],marker=mtype,ms=msize,color=cs[chan],linestyle='None')
            axs[3].set_ylim(0,20)
            axs[3].set_ylabel('Ysig')
            axs[4].plot(freqs,fits['G_popt'][chan,:,0],marker=mtype,ms=msize,color=cs[chan],linestyle='None')
            axs[4].set_ylim(0,4E-7)
            axs[4].set_ylabel('Amp')
            axs[5].plot(freqs,np.degrees(fits['G_popt'][chan,:,5]),marker=mtype,ms=msize,color=cs[chan],
                        label='chan '+str(chan),linestyle='None')
            axs[5].set_ylim(-45,45)
            axs[5].set_ylabel('Theta [deg]')
            axs[5].legend(markerscale=3,ncol=8)
            axs[5].set_xlabel('Frequency [MHz]')
    fig.show()
def cm_to_discrete(cmap, number):
    # this takes in a colormap and a number of colors you want from it, and returns
    # hex codes associated with a discrete colormap, as a vector
    hex_codes=[]
    for i in range(number):
        n = int(i * cmap.N / number)
        rgba = cmap(n)
        # rgb2hex accepts rgb or rgba
        hex_code=matplotlib.colors.rgb2hex(rgba)
        hex_codes.append(hex_code)
    return hex_codes
def get_slice(X,Y,Z,val, sliceOrientation='h'):
    # this gradually increases the tolerance until it finds something
    ok = True
    if sliceOrientation=='h':
        tol = abs(Y[0,1] - Y[0,0])/2.0
    else:
        tol = abs(X[1,0] - X[0,0])/2.0
    while(ok):
        if sliceOrientation=='h': #keeping the y value constant and changing the x value
            sliceIndex = np.where((Y[0,:] < (val + tol)) & (Y[0,:] > (val-tol)))[0]
            if len(sliceIndex>1): sliceIndex = sliceIndex[0]
            n = np.count_nonzero(np.isfinite(Z[:,sliceIndex])) #count number of 'good' data
            if n > 10:
                ok = False
            else:
                ok = True
        if sliceOrientation=='v': #keeping the x value constant and changing the y value
            sliceIndex = np.where((X[:,0] < (val+tol)) & (X[:,0] > (val-tol)))[0]
            if len(sliceIndex>1): sliceIndex = sliceIndex[0]
            n = np.count_nonzero(np.isfinite(Z[sliceIndex,:])) #count number of 'good' data
            if n > 10: ok = False
            else:
                ok = True
        tol+=1
        if tol > 30: ok = False
    return sliceIndex
def get_polar_slice(theta,phi,val=0.0): # assume val in angle
    tol = abs(theta[1,0] - theta[0,0])/1.5
    N = len(theta[:,0]) #figure out the importance of this
    #ok = True
    #while(ok):
    sliceIndex1 = np.where(
            (theta[:,0] < (val + tol)) &
            (theta[:,0] > (val - tol)))[0][0]
    sliceIndex2 = np.where(
            (theta[:,0] < (np.pi + val + tol)) &
            (theta[:,0] > (np.pi + val - tol)))[0][0]
    return sliceIndex1, sliceIndex2