# _   _                       _   _ _                   
#| | (_)                     | | (_) |                  
#| |_ _ _ __ ___   ___  _   _| |_ _| |___   _ __  _   _ 
#| __| | '_ ` _ \ / _ \| | | | __| | / __| | '_ \| | | |
#| |_| | | | | | |  __/| |_| | |_| | \__ \_| |_) | |_| |
# \__|_|_| |_| |_|\___| \__,_|\__|_|_|___(_) .__/ \__, |
#                   ______                 | |     __/ |
#                  |______|                |_|    |___/ 

## 20211110 - WT - creating a new file to contain time fitting functions:

import datetime, pandas,os,h5py,pytz
from scipy.signal import square
from scipy.stats import pearsonr
import numpy as np


## Annie's function for fixing the time axis:
    # (9/28/2021) function for adding sub-second accuracy to DJI timestamps
    # (8/24/2022) bugfix for new datcon changing 'gpsUsed' to 'osd_data:gpsUsed' and 'offsetTime' to 'Clock:offsetTime'
    # now detects and eliminates >1s errors
def interp_time(df_in):
    # find where the GPS turns on
    if 'gpsUsed' in df_in.columns:
        gps_idx = df_in[df_in.gpsUsed == True].index[0]
    ## WT:20220824 bugfix loop:
    elif 'osd_data:gpsUsed' in df_in.columns:
        gps_idx = df_in[df_in['osd_data:gpsUsed'] == True].index[0]
    # interpolate the time and see if it works out!
    while (gps_idx < len(df_in)):
        # look for where the datetimestamp ticks
        first_dts = df_in["GPS:dateTimeStamp"][gps_idx]
        start_sec = int(first_dts[-3:-1])
        while(int(df_in["GPS:dateTimeStamp"][gps_idx][-3:-1]) == start_sec):
            gps_idx = gps_idx + 1
        # use this reference timestamp to convert the offsetTime column into proper datetimes
        start_dt = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx])
        if 'offsetTime' in df_in.columns:
            offsets = np.array(df_in["offsetTime"]-df_in["offsetTime"][gps_idx])
        elif 'Clock:offsetTime' in df_in.columns:
            offsets = np.array(df_in["Clock:offsetTime"]-df_in["Clock:offsetTime"][gps_idx])
        offsets = pandas.to_timedelta(offsets, unit='s')
        timestamps = start_dt + offsets
        # put them in the dataframe
        df_in = df_in.assign(timestamp = timestamps)
        df_in = df_in.assign(UTC = timestamps)
        # check for excessive error by comparing the interpolated and uninterpolated timestamp columns
        gps_dts = pandas.to_datetime(df_in["GPS:dateTimeStamp"][gps_idx:-20]).values
        interp_dts = pandas.to_datetime(df_in["timestamp"][gps_idx:-20]).values
        if (np.mean(np.abs(gps_dts - interp_dts)/np.timedelta64(1,'ms')) < 1000):
            print("Timestamp interpolation succeeded")
            break
        else:
            print("Detected >1s error, retrying")
            gps_idx += 10 # increment the start timestamp index by an arbitrary amount and retry
    return df_in

## source switching signal functions:

def Pulsed_Data_Waveform(total_duration,period,duty_cycle_on):
    ## Outputs should be an array of timedeltas and an array of switch voltages (1s and 0s)
    ## Let's make the time resolution of these arrays milliseconds (10^-3 sec):
    t_steps_ms=int(datetime.timedelta(seconds=total_duration).total_seconds()*1e3)+1 #n_steps
    t_arr_s=np.linspace(0,total_duration,t_steps_ms)
    ## Use the square function from scipy.signal to produce the 1s and 0s:
    switch_signal_arr=0.5*square((2*np.pi/(period*1e-6))*t_arr_s,duty_cycle_on/period)+0.5
    ## Create a timedelta array for interpolation purposes so we can interpolate the square wave later:
    t_arr_datetime=np.array([datetime.timedelta(seconds=timeval) for timeval in t_arr_s])
    return t_arr_s,t_arr_datetime,switch_signal_arr


def Find_File_And_Next(est_time_str, base_path, use_dst=True):
    """
    Given a time in EST/EDT, return:
      - the file containing that time (if any)
      - the next file in sequence (if it exists)
    """
    # EST/EDT offset
    offset_hours = -4 if use_dst else -5
    est_offset = datetime.timedelta(hours=offset_hours)

    # parse target time
    est_time = datetime.datetime.strptime(est_time_str, "%Y-%m-%d %H:%M:%S")
    est_time = est_time.replace(tzinfo=datetime.timezone(est_offset))
    utc_target = est_time.astimezone(datetime.timezone.utc)

    # gather sorted files
    results = []
    for fname in sorted(os.listdir(base_path)):
        if not fname.isdigit() or len(fname) != 4:
            continue
        fpath = os.path.join(base_path, fname)
        try:
            with h5py.File(fpath, 'r') as fd:
                ctimes = fd['index_map']['time']['ctime'][:]
                t0, t1 = ctimes[0], ctimes[-1]
        except Exception:
            continue

        utc_start = datetime.datetime.fromtimestamp(t0, tz=datetime.timezone.utc)
        utc_end   = datetime.datetime.fromtimestamp(t1, tz=datetime.timezone.utc)

        results.append({
            "file": int(fname),
            "UTC_start": utc_start,
            "UTC_end": utc_end,
            "EST_start": utc_start + est_offset,
            "EST_end": utc_end + est_offset
        })

    # make sure sorted by file number
    results = sorted(results, key=lambda r: r["file"])

    containing, next_file = None, None
    for i, r in enumerate(results):
        if r["UTC_start"] <= utc_target <= r["UTC_end"]:
            containing = r
            next_file = results[i+1] if i+1 < len(results) else None
            break
        elif r["UTC_end"] < utc_target:
            containing = r  # keep updating until we pass the target
        elif r["UTC_start"] > utc_target:
            next_file = r
            break

    return containing, next_file

def Get_File_Times(file_number, base_path, use_dst=True):
    """
    Given a 4-digit file number, open the correlator HDF5 file and return 
    the UTC and EST/EDT start/end times.
    
    Parameters
    ----------
    file_number : int or str
        The file number (e.g. 5, 202, or '0202').
    base_path : str
        Path to the directory containing the correlator files 
        (everything up to but not including the file number).
    use_dst : bool
        If True, adjust to Eastern Daylight Time (UTC-4) in summer months,
        otherwise keep fixed at EST (UTC-5).
    
    Returns
    -------
    dict
        Dictionary with UTC and EST/EDT start/end times as datetime objects.
    """
    # normalize file number into zero-padded 4-digit string
    fname = str(file_number).zfill(4)
    fpath = os.path.join(base_path, fname)
    
    with h5py.File(fpath, 'r') as fd:
        ctimes = fd['index_map']['time']['ctime'][:]
    
    # get first and last ctime
    t0 = ctimes[0]
    t_end = ctimes[-1]
    
    # convert to datetime (UTC)
    utc_start = datetime.datetime.fromtimestamp(t0, tz=datetime.timezone.utc)
    utc_end   = datetime.datetime.fromtimestamp(t_end, tz=datetime.timezone.utc)
    
    # Eastern offset: UTC-4 (daylight) or UTC-5 (standard)
    offset_hours = -4 if use_dst else -5
    est_offset = datetime.timedelta(hours=offset_hours)
    
    est_start = utc_start + est_offset
    est_end   = utc_end + est_offset
    
    return {
        "UTC_start": utc_start,
        "UTC_end": utc_end,
        "EST_start": est_start,
        "EST_end": est_end,
    }