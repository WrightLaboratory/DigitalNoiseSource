#  __ _ _   _   _                      _   _ _
# / _(_) | | | (_)                    | | (_) |
#| |_ _| |_| |_ _ _ __   __ _    _   _| |_ _| |___   _ __  _   _
#|  _| | __| __| | '_ \ / _` |  | | | | __| | / __| | '_ \| | | |
#| | | | |_| |_| | | | | (_| |  | |_| | |_| | \__ \_| |_) | |_| |
#|_| |_|\__|\__|_|_| |_|\__, |   \__,_|\__|_|_|___(_) .__/ \__, |
#                        __/ |_____                 | |     __/ |
#                       |___/______|                |_|    |___/
## 20211110 - WT - creating a new file to contain fit functions...
## relocating functions from drone, concat, etc scripts:
## 07/30 - Generalized Fit_Main_Beam() so it works for any number of input
##         channels (not just even/paired counts): channels with no associated
##         dish (NaN dish_coords) are now skipped gracefully (NaN-filled result)
##         instead of raising/propagating garbage fits. Added an optional
##         tbounds input for extra time-index cropping on top of whatever
##         cropping is already baked into inputconcat. Also widened the
##         exception guard slightly (ValueError/IndexError) so a single bad
##         channel/frequency can't silently poison neighboring results.
## 08/14 - coordbounds x/y entries in Fit_Main_Beam() now accept either a
##         scalar (old behavior: symmetric +/-v window about the LC origin)
##         or a (lo,hi) pair for an asymmetric search window. Useful when a
##         previous fit/verification plot shows the true beam center sits
##         off to one side of nominal -- widening the old symmetric box to
##         reach it also pulled in an equal amount of extra low-SNR/
##         background samples on the far side. Fully backward compatible:
##         every existing caller still passes scalars.
## 08/19 - Fixed a parameter-scale conditioning bug in Fit_Main_Beam(): the
##         Gauss/Airy amplitude and background parameters were fit in the
##         input data's raw (absolute) units alongside position/sigma
##         parameters that live on a totally different scale (tens of
##         meters). This worked "well enough" when V was raw ADU^2
##         (~1e-6..1e-9), but once gain-calibrated data pushed
##         amplitude/background down several more orders of magnitude
##         (~1e-15), scipy.optimize.least_squares' default finite-difference
##         step size (which floors at max(1, |x0|), i.e. anchored to O(1))
##         became enormous relative to the actual amplitude/background
##         values, so the numerical Jacobian -- and therefore the fit --
##         stopped tracking the true tiny amplitude at all. Fix: each
##         (channel, freq) fit now non-dimensionalizes V by its own local
##         amp0 before fitting (amp starts at exactly 1.0, background at
##         bg0/amp0), fits entirely in those O(1) units, and rescales
##         amp/bg back afterward. This makes the fit's behavior invariant to
##         whatever absolute units/scale V happens to be in -- raw ADU^2,
##         gain-calibrated, or anything else -- since the optimizer never
##         sees the absolute scale at all. x_scale='jac' is also now passed
##         to least_squares as cheap additional conditioning insurance for
##         the remaining scale gap between amplitude/background (~O(1) after
##         normalization) and position/sigma (~O(10s of meters)).
## 08/21 - Unified auto/cross channel model (matches new corr.py/concat.py):
##         chan now means a flat product index, not a physical port index, so
##         Fit_Main_Beam now reads inputconcat.chan_dish_coords[chan] (the
##         per-channel resolved coordinate -- correct for auto AND cross
##         channels) instead of inputconcat.dish_coords[chan] (port-indexed,
##         would silently misindex once chans includes cross products). Also
##         now reads V/V_bgsub via np.abs() uniformly, since a channel can be
##         genuinely complex regardless of Vargs -- previously this compared/
##         fit raw (possibly complex) values directly.
## 09/16 - Added an optional copol_angle_deg parameter to Fit_Main_Beam(). The
##         2D Gaussian's xsig/ysig (and therefore the FWHM you get out of them)
##         have always implicitly assumed the instrument's copolarization axes
##         line up with the standard local-Cartesian x/y (NS/EW) frame -- true
##         for every flight so far, but not for a dish mounted with its
##         polarizations along NW/NE instead. Rather than touch the fit
##         functions themselves, Fit_Main_Beam now just re-expresses the drone
##         x/y positions (and the dish-position initial guess) in a frame
##         rotated by copol_angle_deg before handing them to the *same*
##         axis-aligned Gauss_2d_LC_func/Airy_2d_LC_func used before -- so
##         xsig/ysig in the returned G_popt are always the FWHM along
##         whatever axes you asked for, with copol_angle_deg=0.0 (the default)
##         reproducing the exact old NS/EW behavior for every existing caller.
##         See _rotate_xy() docstring for the sign convention, and the
##         Synchronization_Function/Synchronization_Verification_Plots
##         changelog notes for how to wire this through for a rotated flight.
import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import least_squares
from astropy.modeling.models import AiryDisk2D
from scipy.stats import pearsonr
## DEFN the Gauss Fit function:
def Gauss(x,a,x0,sigma,k):
    return a*np.exp(-(x-x0)**2.0/(2.0*sigma**2.0))+k
def Airy_2d_LC_opt(P,x,y,V):
    AD=AiryDisk2D(1,5,5,radius=1)
    amp,x0,y0,rad,c=P
    return AD.evaluate(x,y,amp,x0,y0,rad)+c-V
def Airy_2d_LC_func(P,x,y):
    AD=AiryDisk2D(1,5,5,radius=1)
    amp,x0,y0,rad,c=P
    return AD.evaluate(x,y,amp,x0,y0,rad)+c
def Gauss_2d_LC_opt_wtheta(P,x,y,V):
    Gauss_eval = Gauss_2d_LC_func_wtheta(P,x,y)
    return Gauss_eval-V
def Gauss_2d_LC_func_wtheta(P,x,y):
    amp,x0,xsig,y0,ysig,c,theta=P
    A=(0.5*(((np.cos(theta)/xsig)**2.0)+((np.sin(theta)/ysig)**2.0)))
    B=(0.25*((np.sin(2.0*theta)/(ysig**2.0))-(np.sin(2.0*theta)/(xsig**2.0))))
    C=(0.5*(((np.sin(theta)/xsig)**2.0)+((np.cos(theta)/ysig)**2.0)))
    return amp*np.exp(-1.0*((A*((x-x0)**2.0))+(2.0*B*(x-x0)*(y-y0))+(C*((y-y0)**2.0))))+c
def Gauss_2d_LC_opt(P,x,y,V):
    Gauss_eval = Gauss_2d_LC_func(P,x,y)
    return Gauss_eval-V
def Gauss_2d_LC_func(P,x,y):
    amp,x0,xsig,y0,ysig,c=P
    xx = ((x-x0)**2)/(2*(xsig**2))
    yy = ((y-y0)**2)/(2*(ysig**2))
    return amp*np.exp(-1.0*(xx + yy))+c
def _rotate_xy(x,y,angle_deg):
    """
    Re-express local-Cartesian (x,y) coordinates in a frame whose x'-axis
    sits `angle_deg` degrees counterclockwise from the standard x-axis
    (a passive/axis rotation -- this returns each point's coordinates AS SEEN
    FROM the rotated frame, it does not move the points themselves):
        x' =  x*cos(angle) + y*sin(angle)
        y' = -x*sin(angle) + y*cos(angle)
    angle_deg=0.0 is a no-op (returns x,y unchanged).
    Use this to re-express drone positions along the true copolarization
    axes when they are not aligned with the standard NS/EW (x/y)
    local-Cartesian frame -- e.g. a dish mounted with its polarizations
    along NW/NE instead of N/E, which is a 45-degree rotation. The sign
    (+45 vs -45) depends on which physical direction your +x/+y axes point
    and which copol channel you're calling "x" -- there's no way to know
    that from the coordinates alone, so pick a sign, run
    Synchronization_Verification_Plots, and check that the resulting beam
    looks like a clean, centered main lobe along the plotted X'/Y' axes;
    flip the sign if it instead looks skewed/offset.
    """
    if angle_deg==0.0:
        return x,y
    theta=np.radians(angle_deg)
    c,s=np.cos(theta),np.sin(theta)
    xr=x*c+y*s
    yr=-x*s+y*c
    return xr,yr
def Fit_Main_Beam(inputconcat,chans,freqs,theta_solve,coordbounds=[50.0,50.0,150.0],ampbound=0.999,Vargs='None',t_bounds=None,copol_angle_deg=0.0):
    """
    Fit an Airy disk and a 2D Gaussian to the main beam response for every
    requested (channel, frequency) pair.
    Generalized to work for any number of channels -- chans can be any length
    (including 1), does not need to be an even number, and does not need to
    come in X/Y polarization pairs. Any channel whose chan_dish_coords entry
    is NaN (i.e. a channel with no dish/port assigned to it) is skipped and
    its outputs are left as NaN, rather than being fit with garbage
    coordinates. `chan` is a flat product index (0..n_channels-1) -- every
    channel, auto or cross, is fit uniformly via np.abs(V).
    Parameters
    ----------
    coordbounds : [x, y, z]
        x and y entries each accept either:
          - a scalar `v`, giving the old symmetric window (-v, v) about the
            local-Cartesian origin (backwards compatible -- this is what
            every existing caller passes today), or
          - a 2-element (lo, hi) sequence, e.g. (-50.0, 10.0), giving an
            asymmetric window. Use this when the true beam center is known
            (from a previous fit/verification plot) to sit off to one side
            of nominal (0,0): widening a symmetric box to reach it also
            drags in extra low-SNR/background samples on the opposite side,
            which dilutes rather than helps the fit, whereas an asymmetric
            box captures the offset mainlobe without the extra baggage.
        z is unchanged: a scalar far-field/altitude floor, samples are kept
        where |z| > coordbounds[2].
    t_bounds : optional [start_index, stop_index]
        Extra time-index cropping applied on top of whatever cropping
        (t_bounds/t_drone_offset) was already applied when inputconcat was
        constructed. Use this to restrict the fit to a sub-window without
        having to rebuild inputconcat. Defaults to None (no extra cropping).
    copol_angle_deg : float, default 0.0
        Rotation (degrees, see _rotate_xy() for sign convention) of the
        instrument's true copolarization axes away from the standard
        NS/EW local-Cartesian x/y frame. Every existing call implicitly
        assumed 0.0 (copol axes aligned with x/y), which is still the
        default. Set this (e.g. 45.0 for a dish mounted NW/NE instead of
        N/E) to fit -- and therefore report xsig/ysig FWHM -- along the
        actual copolarization axes instead of plain x/y. Only the drone
        x/y positions fed to the fit (and the dish-position initial guess)
        are rotated; coordbounds selection below is still evaluated in the
        original, unrotated x/y frame, since it's just a coarse spatial
        box around the dish and doesn't need to track the copol axes.

    Notes on fit scaling
    --------------------
    Each (channel, freq) fit is performed in units normalized by that
    fit's own local amplitude (amp0 = np.nanmax(mbV)): the data fed to
    least_squares is mbV/amp0, the amplitude parameter starts at exactly
    1.0, and the background parameter starts at bg0/amp0. Fitted
    amplitude/background are rescaled back to the input data's units
    afterward (popt/A_popt/G_popt are all returned in the same units as
    inputconcat.V, unchanged from before). This keeps the fit's numerical
    behavior identical regardless of the absolute scale of V -- raw ADU^2,
    gain-calibrated, or anything else -- since position/sigma parameters
    live on a fixed physical scale (meters) that never moves, while
    amplitude/background otherwise would have tracked whatever arbitrary
    units V happens to be in that call.
    """
    A_popt=np.nan*np.ones((len(chans),len(freqs),5))
    A_PR=np.nan*np.ones((len(chans),len(freqs)))
    G_popt=np.nan*np.ones((len(chans),len(freqs),7))
    G_PR=np.nan*np.ones((len(chans),len(freqs)))
    ## define timecuts for cartesian coordinates. Each of the x/y entries in
    ## coordbounds may be either a scalar (symmetric +/-v window, matching
    ## the historical behavior exactly) or a (lo,hi) pair (asymmetric
    ## window) -- see docstring above:
    def _axis_bounds(entry):
        if np.isscalar(entry):
            return (-abs(entry), abs(entry))
        lo, hi = entry
        return (lo, hi)
    xlo, xhi = _axis_bounds(coordbounds[0])
    ylo, yhi = _axis_bounds(coordbounds[1])
    txcut=inputconcat.t_index[(inputconcat.drone_xyz_LC_interp[:,0]>xlo)&(inputconcat.drone_xyz_LC_interp[:,0]<xhi)]
    tycut=inputconcat.t_index[(inputconcat.drone_xyz_LC_interp[:,1]>ylo)&(inputconcat.drone_xyz_LC_interp[:,1]<yhi)]
    tzcut=inputconcat.t_index[np.abs(inputconcat.drone_xyz_LC_interp[:,2])>coordbounds[2]]
    coordcut=np.intersect1d(np.intersect1d(txcut,tycut),tzcut)
    ## Apply any extra user-specified time-index cropping on top of the above:
    if t_bounds is not None:
        tbcut=inputconcat.t_index[t_bounds[0]:t_bounds[1]]
        coordcut=np.intersect1d(coordcut,tbcut)
    for i,chan in enumerate(chans):
        ## Skip channels that don't exist on this concat object, or that have
        ## no associated dish (NaN dish coordinates) -- there is nothing
        ## meaningful to spatially fit for these, so leave them as NaN:
        if chan>=inputconcat.chan_dish_coords.shape[0] or np.any(np.isnan(inputconcat.chan_dish_coords[chan,0:2])):
            continue
        for j,find in enumerate(freqs):
            try:
                ## Every channel is already a fully-formed correlation product
                ## (auto or cross) -- np.abs() is the uniform, correct way to
                ## get its magnitude regardless of which type it is:
                if Vargs=='bgsub':
                    V_mag_full=np.abs(inputconcat.V_bgsub[:,find,chan])
                else:
                    V_mag_full=np.abs(inputconcat.V[:,find,chan])
                ## apply amplitude cut:
                tacut=inputconcat.t_index[V_mag_full<ampbound*(np.nanmax(V_mag_full))]
                try:
                    ttcut=np.intersect1d(np.intersect1d(coordcut,tacut),inputconcat.inds_on)
                except AttributeError:
                    ttcut=np.intersect1d(coordcut,tacut)
                ## pull the proper coords for fitting:
                mbx=inputconcat.drone_xyz_LC_interp[ttcut,0]
                mby=inputconcat.drone_xyz_LC_interp[ttcut,1]
                mbz=inputconcat.drone_xyz_LC_interp[ttcut,2]
                mbV=V_mag_full[ttcut]
                ## shared params:
                amp0=np.nanmax(mbV)
                bg0=np.nanmin(mbV)
                x00=inputconcat.chan_dish_coords[chan,0]
                y00=inputconcat.chan_dish_coords[chan,1]
                ## Re-express the fit data and the dish-position initial guess
                ## along the true copolarization axes when they differ from
                ## the standard x/y (NS/EW) frame -- see copol_angle_deg
                ## docstring above and _rotate_xy(). copol_angle_deg=0.0 is a
                ## no-op, so every existing caller is unaffected. Everything
                ## below (pA/pG initial guesses, the least_squares calls, and
                ## the pearsonr checks) already reads mbx/mby/x00/y00, so
                ## rotating them here is the only change needed -- the
                ## returned xsig/ysig (and x0/y0) are now in the rotated
                ## (copol) frame, exactly like they'd be in the standard
                ## frame when copol_angle_deg=0.0:
                mbx,mby=_rotate_xy(mbx,mby,copol_angle_deg)
                x00,y00=_rotate_xy(x00,y00,copol_angle_deg)
                ## Non-dimensionalize the data by its own local amplitude
                ## before fitting, so amplitude/background always start at
                ## O(1) regardless of V's absolute units (raw ADU^2,
                ## gain-calibrated, or anything else). This keeps
                ## amplitude/background on a comparable scale to
                ## position/sigma (which live on a fixed physical scale in
                ## meters), avoiding the scipy least_squares default
                ## finite-difference step (floored at max(1,|x0|)) blowing
                ## up relative to a tiny raw amplitude:
                norm = amp0 if (amp0 is not None and not np.isnan(amp0) and amp0 != 0) else 1.0
                mbV_n = mbV/norm
                amp0_n = amp0/norm  # == 1.0 (unless amp0==0, then norm=1 and amp0_n=0)
                bg0_n = bg0/norm
                mb_input_data_n=np.array([mbx,mby,mbV_n])
                ## airy params:
                rad0=25.0
                ## 2dgauss params:
                xsig0=10.0
                ysig0=10.0
                theta0=0.0
                if theta_solve: # if this is true, solve for theta
                    ## initial guess and bounds (amp/c normalized, same as
                    ## the non-theta branch below):
                    pA=np.array([amp0_n,x00,y00,rad0,bg0_n])
                    pG=np.array([amp0_n,x00,xsig0,y00,ysig0,bg0_n,theta0])
                    bnds = ((-np.inf, -np.inf,0, -np.inf,0, -np.pi,-np.inf/4),
                        (np.inf, np.inf,np.inf, np.inf,np.inf, np.pi,np.inf/4))
                    ## run the fits:
                    gres=least_squares(Gauss_2d_LC_opt_wtheta,x0=pG,bounds=bnds,method='trf',x_scale='jac',args=mb_input_data_n).x
                    gpopt=gres.copy()
                    gpopt[0]*=norm  # amp
                    gpopt[5]*=norm  # c (background)
                    G_popt[i,j]=gpopt
                    G_PR[i,j]=pearsonr(mbV,Gauss_2d_LC_func_wtheta(G_popt[i,j],mbx,mby))[0]
                else: # default: don't solve for theta
                    pA=np.array([amp0_n,x00,y00,rad0,bg0_n])
                    pG=np.array([amp0_n,x00,xsig0,y00,ysig0,bg0_n])
                    ## run the fits:
                    gres=least_squares(Gauss_2d_LC_opt,x0=pG,method='trf',x_scale='jac',args=mb_input_data_n).x
                    gpopt=gres.copy()
                    gpopt[0]*=norm  # amp
                    gpopt[5]*=norm  # c (background)
                    G_popt[i,j,0:6]=gpopt
                    G_PR[i,j]=pearsonr(mbV,Gauss_2d_LC_func(G_popt[i,j,0:6],mbx,mby))[0]
                ares=least_squares(Airy_2d_LC_opt,x0=pA,method='trf',x_scale='jac',args=mb_input_data_n).x
                apopt=ares.copy()
                apopt[0]*=norm  # amp
                apopt[4]*=norm  # c (background)
                A_popt[i,j]=apopt
                A_PR[i,j]=pearsonr(mbV,Airy_2d_LC_func(A_popt[i,j,:],mbx,mby))[0]
            except (ValueError, IndexError):
                A_popt[i,j,:]=np.nan*np.zeros(5)
                A_PR[i,j]=np.nan
                G_popt[i,j,:]=np.nan*np.zeros(7)
                G_PR[i,j]=np.nan
    return A_popt,A_PR,G_popt,G_PR