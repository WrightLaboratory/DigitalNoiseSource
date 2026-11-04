# _____ _________________
#/  __ \  _  | ___ \ ___ \
#| /  \/ | | | |_/ / |_/ /  _ __  _   _
#| |   | | | |    /|    /  | '_ \| | | |
#| \__/\ \_/ / |\ \| |\ \ _| |_) | |_| |
# \____/\___/\_| \_\_| \_(_) .__/ \__, |
#                          | |     __/ |
#                          |_|    |___/
##
## 2026-11 Correlator data class for ICEBoard and CRS data.
##
## USER INPUT IS ONLY:
##   * a site class file (site_utils.Site): which ports had something on them
##     (chmap) and what it was (keystrings)
##   * which instrument wrote the data: instrument='ice' (default) or 'crs'
##
##   corrdata = corr.Corr_Data(Data_Directory=datadir, Gain_Directory=gaindir,
##                             site_class=gbosite, instrument='crs',
##                             fbounds=[0,8192], Load_Gains=True, Apply_Gains=False,
##                             remove_real_imag_offset=True,
##                             Data_File_Index=[0,-1], offset_time_samples=100)
##
## The class knows each instrument's raw product ordering, finds every auto
## and cross between the used ports, and names every channel from the site
## file keystrings:
##
##   corrdata.V        all channels (time, freq, channel), complex
##   corrdata.V_auto   autos only     (its own 0..n_auto-1 indexing)
##   corrdata.V_cross  crosses only   (its own 0..n_cross-1 indexing)
##
##   corrdata.print_channels()     table: V index | V_auto / V_cross index | name | ports | ...
##   corrdata.channel_table        same table as a pandas DataFrame
##   print(corrdata.chnames)       0: Ref(1) x Ref(1) / 1: Ref(1) x Dish(3) / ...
##   print(corrdata.cross_names)   the same, numbered as in V_cross
##   corrdata.chnames[1]                  -> 'Ref(1) x Dish(3)'
##   corrdata.chnames['Ref(1) x Dish(3)'] -> 1        (index in V)
##   corrdata.cross_names['Ref(1) x Dish(3)'] -> 0    (index in V_cross)
##   corrdata.chan_index('Ref', 'Dish')               -> 1  (keystrings, 'Ref(1)', or port numbers)
##   corrdata.chan_index('Ref', 'Dish', array='V_cross') -> 0
##   corrdata.get_chan('Dish', 'Ref')     data in exactly that order (conjugated if needed)
##   corrdata.magnitude('Ref', 'Dish')    np.abs() of the channel
##
## AUTO vs CROSS:
##   * auto  (A x A): power, imaginary part ~0, so np.abs(V) == V.real.
##   * cross (A x B): complex (voltage-product) - magnitude is ONLY np.abs(V);
##     V.real is just one quadrature component.
##   Use np.abs() everywhere; it is correct for both. Channel names say which is
##   which (A x A = auto). Every name reads exactly as the data is stored: if the
##   instrument saved B x A but chmap order makes the name A x B, the data was
##   conjugated on load so the name is literally true.
##
## GAINS (same recipe as the original corr.py):
##   Both boards multiply each INPUT's channelised voltage by a digital gain g
##   (per input, per frequency) before correlating. A product of inputs a and b
##   therefore comes out scaled by g_a * g_b, so dividing every product - auto
##   or cross - by g_a * g_b undoes it. One gain per input is all that's needed.
##   ICEBoard: .h5 file, g = gain_coeff[0] * 2**gain_exp[0], all 16 inputs.
##   CRS:      .npy file, (8, 8192), stored in FPGA bit-reversed bin order,
##             descrambled here. In both cases the real part of g is used and
##             V_ab /= g_a * g_b, exactly as the original.
##
## TIME:
##   corrdata.t              UNIX seconds of every integration
##   corrdata.t_rel          seconds since the first loaded integration
##   corrdata.t_arr_datetime UTC datetimes
##   ICEBoard default (time_source='fpga', as the original GBO code): FPGA
##   counts (2.56 us each) relative to the first loaded integration, anchored
##   to that integration's IRIG-B time (or ctime if use_ctime=True).
##   time_source='irigb' or 'ctime' uses those per-integration stamps directly
##   (what the old code did for D3A / CHIME). CRS: stamps are already UNIX s.
##
## No behaviour depends on the site name or the data file names.

from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np
import h5py
import hdf5plugin
import os
import re
import glob
from matplotlib import colors
import pandas
import csv
import datetime
import pytz


############################################################
# INSTRUMENTS AND PRODUCT MAPS
############################################################
INSTRUMENT_ALIASES = {'ice': 'ICE', 'iceboard': 'ICE', 'ice_board': 'ICE', 'crs': 'CRS'}
INSTRUMENT_LONGNAME = {'ICE': 'ICEBoard', 'CRS': 'CRS board'}
FPGA_SECONDS_PER_COUNT = 2.56e-6

## CRS: the board always writes all 36 products of its 8 inputs (you can't
## select fewer), but index_map/prod isn't populated correctly, so the order is
## defined here. Covers all 8 autos + 28 crosses exactly once.
CRS_PROD = np.array([
    [0,0],[0,1],[0,2],[0,3],
    [1,1],[1,2],[1,3],[1,4],
    [2,2],[2,3],[2,4],[2,5],
    [3,3],[3,4],[3,5],[3,6],
    [4,4],[4,5],[4,6],[4,7],
    [5,5],[5,6],[5,7],[0,4],
    [6,6],[6,7],[0,5],[1,5],
    [7,7],[0,6],[1,6],[2,6],
    [0,7],[1,7],[2,7],[3,7],
], dtype=int)


def ice_prod_map(n_inputs):
    """ICEBoard product order for inputs 0..n_inputs-1 (the board always records
    the first N inputs, in order): upper triangle row by row, autos included,
        (0,0),(0,1),...,(0,N-1),(1,1),...,(N-1,N-1)
    e.g. N=4 -> 10 products; (1,1)=4, (1,3)=6, (3,3)=9."""
    return np.array([[a, b] for a in range(n_inputs) for b in range(a, n_inputs)], dtype=int)


def _n_inputs_from_n_prod(n_prod):
    n = int(round((np.sqrt(8 * n_prod + 1) - 1) / 2))
    return n if n * (n + 1) // 2 == n_prod else None


def _plural(n, word):
    return '{} {}'.format(n, word if n == 1 else (word + 'es' if word.endswith('ss') else word + 's'))


def _to_complex(arr):
    """complex128, handling structured ('r','i') dtypes."""
    arr = np.asarray(arr)
    if arr.dtype.names is not None and 'r' in arr.dtype.names and 'i' in arr.dtype.names:
        return arr['r'].astype(np.float64) + 1j * arr['i'].astype(np.float64)
    return arr.astype(np.complex128)


############################################################
# CHANNEL NAME LIST
############################################################
_X_SPLIT = re.compile(r'\s+[xX×]\s+|(?<=\))\s*[xX×]\s*(?=\S)')


def _norm_name(s):
    return re.sub(r'\s+', '', str(s))


class ChannelNames(list):
    """Channel names that print as a numbered list and index both ways:
        names[1]                  -> 'Ref(1) x Dish(3)'
        names['Ref(1) x Dish(3)'] -> 1   (whitespace-insensitive)
        names[:] / names[[0,2]]   -> numbered sub-list keeping the original indices
    `array` says which array the numbers index ('V', 'V_auto' or 'V_cross')."""
    def __init__(self, names, idx=None, array='V'):
        super().__init__([str(n) for n in names])
        self._idx = list(range(len(self))) if idx is None else [int(i) for i in idx]
        self.array = array

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.index(key)
        if isinstance(key, slice):
            sel = list(range(len(self)))[key]
        elif isinstance(key, (list, tuple, np.ndarray)):
            key = np.asarray(key)
            sel = list(np.where(key)[0]) if key.dtype == bool else [int(k) for k in key]
        else:
            return list.__getitem__(self, key)
        return ChannelNames([list.__getitem__(self, i) for i in sel],
                            idx=[self._idx[i] for i in sel], array=self.array)

    def index(self, name, *args):
        target = _norm_name(name)
        for pos, n in enumerate(self):
            if _norm_name(n) == target:
                return self._idx[pos]
        parts = _X_SPLIT.split(str(name).strip())
        if len(parts) == 2:
            swapped = _norm_name(parts[1] + ' x ' + parts[0])
            for pos, n in enumerate(self):
                if _norm_name(n) == swapped:
                    raise KeyError(
                        "'{}' is stored as '{}' ({} index {}). Your order is its complex "
                        "conjugate - corrdata.get_chan(...) returns it conjugated.".format(
                            name, n, self.array, self._idx[pos]))
        raise KeyError("No channel named '{}' in {}. Available:\n{}".format(
            name, self.array, str(self)))

    def __str__(self):
        if len(self) == 0:
            return '(no channels)'
        w = len(str(max(self._idx)))
        return '\n'.join('{:>{w}}: {}'.format(i, n, w=w) for i, n in zip(self._idx, self))

    __repr__ = __str__


############################################################
# MAIN CLASS
############################################################
class Corr_Data:
    def __init__(self, Data_Directory, Gain_Directory, site_class,
                 instrument=None,
                 Data_File_Index=[0, -1], Load_Gains=True, Fix_Gains=False,
                 Apply_Gains=True, Gain_Params=[1.0, 24.0],
                 fbounds=[0, 1024],
                 use_ctime=False, time_source='fpga',
                 skip_last_file=None,
                 esystem=False,
                 remove_real_imag_offset=False,
                 offset_file_index=0,
                 offset_time_samples=400,
                 **deprecated):
        """
        Data_Directory  folder of correlator .h5 files
        Gain_Directory  ICE: folder holding the gain .h5 (or the .h5 itself)
                        CRS: the .npy gain file for this acquisition
        site_class      site_utils.Site (chmap, keystrings, n_channels)
        instrument      'ice' (default) or 'crs'
        Data_File_Index [start, stop] slice of the file list (default [0,-1]);
                        None = all files; or any other list of file indices
        Load_Gains      read the gain file
        Fix_Gains       use Gain_Params[0] * 2**Gain_Params[1] for every input
                        instead (takes priority over Load_Gains)
        Apply_Gains     divide gains out of the data
        fbounds         [first, last) frequency channels to load
        use_ctime       ICE: anchor the FPGA time axis to ctime, not IRIG-B
        time_source     ICE: 'fpga' (default), 'irigb' or 'ctime' (see header)
        skip_last_file  drop the newest file (may still be being written).
                        Default: True for ICE (as the original), False for CRS
        esystem         CHIME-format eigen-system files: use eigenvalue 0 x |evec 0|^2
        remove_real_imag_offset
                        subtract the median real and imaginary offset (per
                        frequency) from every CROSS channel, estimated from the
                        first offset_time_samples integrations of file number
                        offset_file_index. Autos untouched. Stored as
                        corrdata.offset_real / offset_imag, shape (freq, n_cross),
                        column k = V_cross channel k.
        """
        ## ---- keywords from older versions ----
        crs_flag = deprecated.pop('crs', None)
        notes = {
            'crossmap': "crossmap is no longer needed: every auto and cross between the "
                        "chmap ports is loaded (see corrdata.V_cross / print_channels()).",
            'offset_crossmap': "offset_crossmap is no longer needed: offsets are removed from "
                               "every cross channel.",
            'auto_only': "auto_only was removed: use corrdata.V_auto.",
            'offset_freq_index': "offset_freq_index / plot_offset_diagnostic: the offset plot "
                                 "has moved out of corr (to go in fitting_utils).",
            'plot_offset_diagnostic': None,
        }
        for old, msg in notes.items():
            if old in deprecated:
                deprecated.pop(old)
                if msg:
                    print("  --> NOTE: " + msg)
        if deprecated:
            raise TypeError("Corr_Data got unexpected keyword(s): {}".format(list(deprecated)))

        defaulted = instrument is None and not crs_flag
        if instrument is None:
            instrument = 'crs' if crs_flag else 'ice'
        key = str(instrument).strip().lower()
        if key not in INSTRUMENT_ALIASES:
            raise ValueError("instrument must be 'ice' or 'crs', got {!r}".format(instrument))
        self.instrument = INSTRUMENT_ALIASES[key]
        time_source = str(time_source).lower()
        if time_source not in ('fpga', 'irigb', 'ctime'):
            raise ValueError("time_source must be 'fpga', 'irigb' or 'ctime'")

        self.Data_Directory = Data_Directory
        self.Gain_Directory = Gain_Directory
        self.site_class = site_class
        self.fbounds = list(fbounds)
        self.use_ctime = use_ctime
        self.time_source = time_source
        self.esystem = esystem
        self.Apply_Gains = Apply_Gains
        self._cache = {}

        print('Initializing Correlator Class')
        print("  --> Instrument: {} (instrument='{}'){}".format(
            INSTRUMENT_LONGNAME[self.instrument], self.instrument.lower(),
            '  [not specified -> default]' if defaulted else ''))
        print("  --> Data: " + str(self.Data_Directory))

        ## ---- file list ----
        if skip_last_file is None:
            skip_last_file = self.instrument == 'ICE'
        filelist = self._list_files(skip_last_file)
        self.filenames = self._select_files(filelist, Data_File_Index)
        n_files = len(self.filenames)
        if n_files == 0:
            raise RuntimeError(
                "No data files selected in {}: {} usable file(s) and Data_File_Index={}. "
                "Note [0,-1] leaves out the last file of the list{} - use "
                "Data_File_Index=None for all of them.".format(
                    self.Data_Directory, len(filelist), Data_File_Index,
                    ' (on top of the newest file already skipped)' if skip_last_file else ''))
        print("  --> Loading {} of {} files ({} ... {})".format(
            n_files, len(filelist), self.filenames[0], self.filenames[-1]))

        ## ---- first file: axes, product map, time anchor ----
        with h5py.File(os.path.join(self.Data_Directory, self.filenames[0]), 'r') as fd:
            self._layout = self._detect_layout(fd)
            vis0 = self._read_vis(fd)
            n_time, n_freq, n_prod_raw = vis0.shape
            del vis0
            self.freq, n_freq_full = self._read_freq(fd)
            self._set_time_anchor(fd)
            self.prod = self._product_map(fd, n_prod_raw)
        self.n_freq_full = n_freq_full
        self.n_inputs_hw = int(self.prod.max()) + 1
        print("  --> Raw products per file: {} (from {} inputs)".format(
            len(self.prod), self.n_inputs_hw))

        ## ---- used ports + channel names from the site class ----
        self._setup_ports(site_class)
        self._setup_channels()

        ## ---- gains ----
        self._setup_gains(Load_Gains, Fix_Gains, Gain_Params, n_freq)

        ## ---- offsets (crosses only) ----
        self.offset_real = self.offset_imag = self._offset = None
        if remove_real_imag_offset:
            self._compute_offsets(filelist, offset_file_index, offset_time_samples)

        ## ---- main loop ----
        self.V = np.zeros((n_files, n_time, n_freq, self.n_channels), dtype=complex)
        t = np.zeros((n_files, n_time))
        self.sat = np.zeros((n_files, n_time, n_freq, self.n_ports))
        valid = np.zeros((n_files, n_time), dtype=bool)
        print("Assigning array values by reading in data files:")
        for i, file in enumerate(self.filenames):
            print("\r  --> Loading file {}/{}: {}".format(i + 1, n_files, file), end="")
            try:
                with h5py.File(os.path.join(self.Data_Directory, file), 'r') as fd_n:
                    if fd_n['vis'].shape[0 if self._layout == 'tfp' else 2] == 0:
                        print("\n  --> Skipping empty file: {}".format(file))
                        continue
                    chans, tm, sat = self._read_file(fd_n)
            except (OSError, KeyError) as e:
                print("\n  --> Skipping file {} ({})".format(file, e))
                continue
            nt = chans.shape[0]
            if nt != n_time:
                print("\n  --> File {} has {} integrations (expected {}); keeping {}.".format(
                    file, nt, n_time, min(nt, n_time)))
                nt = min(nt, n_time)
            if self._offset is not None:
                chans[:, :, self.cross_idx] -= self._offset[np.newaxis, :, :]
            self.V[i, :nt] = chans[:nt]
            t[i, :nt] = tm[:nt]
            if sat is not None:
                self.sat[i, :nt] = sat[:nt]
            valid[i, :nt] = True
        print("\n  --> Finished. Reshaping arrays.")

        ## ---- flatten files x time, drop skipped / missing integrations ----
        valid = valid.reshape(-1)
        self.V = self.V.reshape((n_files * n_time, n_freq, self.n_channels))[valid]
        self.t = t.reshape(n_files * n_time)[valid]
        self.sat = self.sat.reshape((n_files * n_time, n_freq, self.n_ports))[valid]
        if (~valid).any():
            print("  --> Dropped {} empty/missing integrations.".format(int((~valid).sum())))
        self.t_rel = self.t - self.t[0] if len(self.t) else self.t
        self.t_arr_datetime = np.array(
            [datetime.datetime.fromtimestamp(x, pytz.utc) for x in self.t])
        self.t_index = np.arange(len(self.t))
        if len(self.t):
            print("  --> Time: {} to {} UTC ({})".format(
                self.t_arr_datetime[0].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                self.t_arr_datetime[-1].strftime('%H:%M:%S.%f')[:-3], self._time_desc))
        print("  --> V shape (time, freq, channel): {}  [{}, {}]".format(
            self.V.shape, _plural(len(self.auto_idx), 'auto'), _plural(len(self.cross_idx), 'cross')))

        self._check_autos()
        print()
        self.print_channels()

    ############################################################
    # FILES AND LAYOUT
    ############################################################
    def _list_files(self, skip_last_file):
        files = [x for x in os.listdir(self.Data_Directory)
                 if ".lock" not in x and not x.startswith('.')
                 and not os.path.isdir(os.path.join(self.Data_Directory, x))]
        if self.instrument == 'CRS':
            num = [x for x in files if x.split('.')[0].isdigit()]
            other = sorted(set(files) - set(num))
            if other:
                print("  --> Ignoring non-numbered files: {}".format(', '.join(other)))
            files = sorted(num, key=lambda x: int(x.split('.')[0]))
        else:
            files = sorted(files)
        if skip_last_file and files:
            print("  --> Skipping newest file {} (may still be being written; "
                  "skip_last_file=False to keep it)".format(files[-1]))
            files = files[:-1]
        return files

    @staticmethod
    def _select_files(filelist, idx):
        if idx is None:
            return list(filelist)
        if isinstance(idx, slice):
            return list(filelist[idx])
        idx = list(np.atleast_1d(idx))
        if len(idx) == 2:
            return list(filelist[idx[0]:idx[1]])
        return [filelist[int(i)] for i in idx]

    def _detect_layout(self, fd):
        """'tfp' (time, freq, product) or 'fpt' (freq, product, time, CHIME-style),
        from the file's own axis lengths."""
        if self.instrument == 'CRS':
            return 'tfp'
        shp = fd['vis'].shape
        n_f = fd['index_map']['freq'].shape[0]
        if shp[1] == n_f:
            return 'tfp'
        if shp[0] == n_f:
            print("  --> vis stored as (freq, prod, time); transposing")
            return 'fpt'
        raise RuntimeError("Can't match vis shape {} to {} frequency channels".format(shp, n_f))

    def _read_vis(self, fd):
        """Raw visibilities as (time, freq, raw product), frequency-sliced."""
        flb, fub = self.fbounds
        if self.esystem:
            Cevals = np.tile(np.array(fd['eval']).transpose(2, 0, 1)[:, :, :, np.newaxis][:, :, 0, :],
                             fd['evec'].shape[2])
            Cevecs = np.abs(np.array(fd['evec'][:, 0, :, :]).transpose(2, 0, 1)) ** 2.0
            return _to_complex(Cevals * Cevecs)[:, flb:fub, :]
        if self._layout == 'fpt':
            return _to_complex(fd['vis'][flb:fub]).transpose(2, 0, 1)
        return _to_complex(fd['vis'][:, flb:fub, :])

    def _read_freq(self, fd):
        flb, fub = self.fbounds
        f = fd['index_map']['freq']
        n_full = f.shape[0]
        if fub > n_full or flb < 0 or flb >= fub:
            raise ValueError("fbounds {} outside the {} channels in the data".format(
                self.fbounds, n_full))
        if f.dtype.names is not None:          # ICE: structured (centre, width)
            return np.array([x[0] for x in f[flb:fub]]), n_full
        return np.array(f[flb:fub]), n_full

    ############################################################
    # TIME
    ############################################################
    def _set_time_anchor(self, fd):
        if self.instrument == 'CRS':
            self._time_desc = 'CRS UNIX timestamps'
            return
        names = fd['index_map']['time'].dtype.names or ()
        src = self.time_source
        if src == 'fpga' and 'fpga_count' not in names:
            print("  --> No fpga_count in file; using irigb_time per integration")
            src = 'irigb'
        need = {'fpga': 'ctime' if self.use_ctime else 'irigb_time',
                'irigb': 'irigb_time', 'ctime': 'ctime'}[src]
        if need not in names:
            raise KeyError("time_source='{}' needs '{}' in index_map/time (has {})".format(
                src, need, names))
        self._time_src = src
        tmap = fd['index_map']['time']
        if src == 'fpga':
            self._fpga0 = float(tmap['fpga_count'][0])
            self._tanchor = float(tmap['ctime'][0]) if self.use_ctime \
                else 1e-9 * float(tmap['irigb_time'][0])
            self._time_desc = 'FPGA counts anchored to {} of first integration'.format(
                'ctime' if self.use_ctime else 'IRIG-B')
        else:
            self._time_desc = '{} per integration'.format(src)
        self.t0 = self._tanchor if src == 'fpga' else None

    def _read_times(self, fd):
        """UNIX seconds for every integration in this file."""
        if self.instrument == 'CRS':
            return np.array(fd['index_map']['time'], dtype=float)
        tmap = fd['index_map']['time']
        if self._time_src == 'fpga':
            return self._tanchor + FPGA_SECONDS_PER_COUNT * (
                np.array(tmap['fpga_count'], dtype=float) - self._fpga0)
        if self._time_src == 'irigb':
            return 1e-9 * np.array(tmap['irigb_time'], dtype=float)
        return np.array(tmap['ctime'], dtype=float)

    ############################################################
    # PRODUCT MAP
    ############################################################
    def _product_map(self, fd, n_prod_raw):
        """(n_prod_raw, 2) physical input pair of each raw product."""
        if self.instrument == 'CRS':
            if n_prod_raw != len(CRS_PROD):
                raise RuntimeError("Expected {} CRS products, but vis has {}".format(
                    len(CRS_PROD), n_prod_raw))
            print("  --> Product map: built-in CRS table (36 products, 8 inputs)")
            return CRS_PROD.copy()
        if self.esystem:
            print("  --> Product map: eigen-system mode, one column per input")
            return np.array([[i, i] for i in range(n_prod_raw)], dtype=int)
        n_in = _n_inputs_from_n_prod(n_prod_raw)
        expected = ice_prod_map(n_in) if n_in is not None else None
        file_prod = None
        try:
            p = fd['index_map']['prod'][:]
            file_prod = np.array([[int(x[0]), int(x[1])] for x in p], dtype=int)
        except KeyError:
            pass
        if file_prod is not None and len(file_prod) == n_prod_raw:
            if expected is not None and np.array_equal(file_prod, expected):
                print("  --> Product map: standard ICEBoard order, inputs 0-{} "
                      "(matches the file's index_map/prod)".format(n_in - 1))
            else:
                print("  --> WARNING: the file's index_map/prod isn't the standard ICEBoard "
                      "order; using the file's map.")
            return file_prod
        if expected is None:
            raise RuntimeError("ICEBoard file has {} products (not N(N+1)/2 for any N) and no "
                               "usable index_map/prod.".format(n_prod_raw))
        print("  --> Product map: standard ICEBoard order, inputs 0-{} "
              "(file has no index_map/prod)".format(n_in - 1))
        return expected

    ############################################################
    # PORTS AND CHANNELS
    ############################################################
    def _setup_ports(self, site_class):
        chmap = np.atleast_1d(np.array(site_class.chmap)).astype(int)
        n_used = getattr(site_class, 'n_channels', None)
        if n_used is not None:
            n_used = int(np.squeeze(n_used))
            if n_used > len(chmap):
                raise ValueError("site_class.n_channels={} but chmap only lists {} ports".format(
                    n_used, len(chmap)))
            chmap = chmap[:n_used]
        if len(set(chmap.tolist())) != len(chmap):
            raise ValueError("site_class.chmap repeats a port: {}".format(chmap.tolist()))
        bad = [int(p) for p in chmap if p < 0 or p >= self.n_inputs_hw]
        if bad:
            raise ValueError("chmap port(s) {} not in this data: the {} recorded inputs 0-{} "
                             "only.".format(bad, INSTRUMENT_LONGNAME[self.instrument],
                                            self.n_inputs_hw - 1))
        self.chmap = chmap
        self.n_ports = len(chmap)
        self.n_dishes = self.n_ports            # as in corr_crs_mod_draft

        keys = None
        for attr in ('keystrings', 'keystring', 'Keystrings', 'KeyStrings', 'key_strings'):
            if hasattr(site_class, attr) and not callable(getattr(site_class, attr)):
                keys = getattr(site_class, attr)
                break
        if keys is None:
            print("  --> WARNING: no keystrings in site class; naming ports 'ch(N)'.")
            keys = ['ch'] * self.n_ports
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in np.atleast_1d(keys)]
        if len(keys) < self.n_ports:
            raise ValueError("site class has {} keystrings for {} used ports".format(
                len(keys), self.n_ports))
        self.keystrings = keys[:self.n_ports]
        self.port_labels = ['{}({})'.format(k, p) for k, p in zip(self.keystrings, self.chmap)]

        # dish of each port, if the site file says (site_utils v2 / inferred for old files)
        self.port_dish = [None] * self.n_ports
        c2d = getattr(site_class, 'chmap_to_dishes', None)
        dn = getattr(site_class, 'dish_names', None)
        if c2d is not None and dn is not None and len(c2d) >= self.n_ports:
            self.port_dish = [str(dn[int(d)]) for d in np.asarray(c2d)[:self.n_ports]]
        print("  --> Used ports: " + ', '.join(self.port_labels))

        self.automap = np.zeros(self.n_ports, dtype=int)     # raw auto product of each port
        for i, p in enumerate(self.chmap):
            m = np.where((self.prod[:, 0] == p) & (self.prod[:, 1] == p))[0]
            if len(m) == 0:
                raise RuntimeError("No auto-correlation product for port {}".format(p))
            self.automap[i] = m[0]

    def _setup_channels(self):
        """Every pair (i <= j) of used ports, in chmap order."""
        names, ports, keys, raw, conj, dishes = [], [], [], [], [], []
        for i in range(self.n_ports):
            for j in range(i, self.n_ports):
                pa, pb = int(self.chmap[i]), int(self.chmap[j])
                fwd = np.where((self.prod[:, 0] == pa) & (self.prod[:, 1] == pb))[0]
                rev = np.where((self.prod[:, 0] == pb) & (self.prod[:, 1] == pa))[0]
                if len(fwd):
                    raw.append(int(fwd[0])); conj.append(False)
                elif len(rev):
                    raw.append(int(rev[0])); conj.append(True)   # stored as pb x pa
                else:
                    raise RuntimeError("No product for ports {} x {}".format(pa, pb))
                names.append('{} x {}'.format(self.port_labels[i], self.port_labels[j]))
                ports.append([pa, pb])
                keys.append([self.keystrings[i], self.keystrings[j]])
                dishes.append([self.port_dish[i], self.port_dish[j]])
        self.chnames = ChannelNames(names, array='V')
        self.chan_ports = np.array(ports, dtype=int)      # physical ports, name order
        self.chan_keys = np.array(keys)                    # keystrings, name order
        self.chan_dishes = dishes
        self.chan_raw_idx = np.array(raw, dtype=int)       # raw product index in the file
        self.chan_conj = np.array(conj, dtype=bool)        # conjugated on load
        self.chan_is_auto = self.chan_ports[:, 0] == self.chan_ports[:, 1]
        self.n_channels = len(names)                       # channels in V (draft meaning)
        self.n_chans = self.n_channels

        ## index maps between V and the auto / cross sub-arrays
        self.auto_idx = np.where(self.chan_is_auto)[0]     # V index of V_auto[k]
        self.cross_idx = np.where(~self.chan_is_auto)[0]   # V index of V_cross[k]
        self.chan_sub_idx = np.zeros(self.n_channels, dtype=int)   # V index -> sub-array index
        self.chan_sub_idx[self.auto_idx] = np.arange(len(self.auto_idx))
        self.chan_sub_idx[self.cross_idx] = np.arange(len(self.cross_idx))
        self.auto_names = ChannelNames([names[k] for k in self.auto_idx], array='V_auto')
        self.cross_names = ChannelNames([names[k] for k in self.cross_idx], array='V_cross')

        rows = []
        for k in range(self.n_channels):
            da, db = dishes[k]
            if self.chan_is_auto[k]:
                typ = 'auto'
            elif da is not None and da == db:
                typ = 'cross, same dish'
            else:
                typ = 'cross'
            rows.append(dict(
                V=k,
                V_auto=int(self.chan_sub_idx[k]) if self.chan_is_auto[k] else None,
                V_cross=None if self.chan_is_auto[k] else int(self.chan_sub_idx[k]),
                name=names[k], type=typ,
                port_a=ports[k][0], port_b=ports[k][1],
                key_a=keys[k][0], key_b=keys[k][1],
                dish_a=da, dish_b=db,
                raw_prod=raw[k], conjugated=conj[k]))
        self.channel_table = pandas.DataFrame(rows).set_index('V')
        for c in ('V_auto', 'V_cross'):
            self.channel_table[c] = self.channel_table[c].astype('Int64')
        ## older-draft aliases
        self.keep_idx = self.chan_raw_idx
        self.chan_prod = self.chan_ports

    ############################################################
    # GAINS
    ############################################################
    def _setup_gains(self, Load_Gains, Fix_Gains, Gain_Params, n_freq):
        """self.gain: (freq, input) real gain, indexed by PHYSICAL input number."""
        self.gain = np.ones((n_freq, self.n_inputs_hw))
        self.gain_coeffs = np.ones(n_freq)
        self.gain_exp = np.ones(self.n_inputs_hw)
        self.gainfile = None
        self.gains_loaded = False
        if Fix_Gains:
            if Load_Gains:
                print("  --> Fix_Gains=True: using fixed gains, gain file not read")
            g = Gain_Params[0] * (2.0 ** Gain_Params[1])
            self.gain = g * np.ones((n_freq, self.n_inputs_hw))
            self.gain_coeffs = Gain_Params[0] * np.ones(n_freq)
            self.gain_exp = Gain_Params[1] * np.ones(self.n_inputs_hw)
            self.gains_loaded = True
            print("  --> Gains: fixed at {} x 2^{} for every input".format(*Gain_Params))
        elif Load_Gains:
            self.gains_loaded = self._load_crs_gains() if self.instrument == 'CRS' \
                else self._load_ice_gains()
        else:
            print("  --> Gains: not loaded (Load_Gains=False)")
        if self.Apply_Gains:
            print("  --> Applying gains: {}".format(
                'yes, V_ab / (g_a * g_b)' if self.gains_loaded
                else 'NO GAINS AVAILABLE - data left in raw correlator units'))
        else:
            print("  --> Applying gains: no (Apply_Gains=False)")

    def _load_ice_gains(self):
        """ICEBoard .h5: g = gain_coeff[0] * 2**gain_exp[0], (n_freq_full, 16)."""
        flb, fub = self.fbounds
        try:
            path = self.Gain_Directory
            if path is None:
                raise OSError("Gain_Directory is None")
            if os.path.isdir(path):
                cands = sorted(x for x in os.listdir(path)
                               if x.lower().endswith(('.h5', '.hdf5')) and not x.startswith('.'))
                if not cands:
                    raise OSError("no .h5 gain file in {}".format(path))
                if len(cands) > 1:
                    print("  --> {} gain files in {}; using {} (give the file path to "
                          "choose)".format(len(cands), path, cands[0]))
                path = os.path.join(path, cands[0])
            self.gainfile = os.path.basename(path)
            with h5py.File(path, 'r') as fg:
                coeff = _to_complex(fg['gain_coeff'][0])          # (n_freq_full, n_inputs)
                exp = np.array(fg['gain_exp'][0], dtype=float)     # (n_inputs,)
            g = coeff * np.power(2.0, exp)[np.newaxis, :]
            if g.shape[0] != self.n_freq_full:
                raise ValueError("gain file has {} frequencies, data has {}".format(
                    g.shape[0], self.n_freq_full))
            if g.shape[1] < self.n_inputs_hw:
                raise ValueError("gain file has {} inputs, data has {}".format(
                    g.shape[1], self.n_inputs_hw))
            if np.max(np.abs(g.imag)) > 1e-6 * max(np.max(np.abs(g.real)), 1e-300):
                print("  --> NOTE: gains have an imaginary part; like the original corr.py, "
                      "only the real part is used.")
            self.gain = g.real[flb:fub, :self.n_inputs_hw]    # first N inputs = recorded ones
            self.gain_coeffs, self.gain_exp = coeff, exp
            print("  --> Gains: loaded {} ({} inputs in file, {} used by the data)".format(
                self.gainfile, g.shape[1], self.n_inputs_hw))
            return True
        except (OSError, KeyError, ValueError, IndexError, TypeError) as e:
            print("  --> ERROR: ICE gain file not loaded ({})".format(e))
            return False

    def _load_crs_gains(self):
        """CRS .npy, (8, n_freq_full) or transposed, in FPGA bit-reversed bin order."""
        flb, fub = self.fbounds
        try:
            if self.Gain_Directory is None or os.path.isdir(self.Gain_Directory):
                raise ValueError("for CRS, Gain_Directory must be the .npy gain file for this "
                                 "acquisition (e.g. gaindir = gdir + '<file>.npy')")
            self.gainfile = os.path.basename(self.Gain_Directory)
            g = np.load(self.Gain_Directory, allow_pickle=True)
            n_full = self.n_freq_full
            if g.shape == (self.n_inputs_hw, n_full):
                g = g.T
            elif g.shape != (n_full, self.n_inputs_hw):
                raise ValueError("CRS gain array has shape {}, expected ({}, {})".format(
                    g.shape, self.n_inputs_hw, n_full))
            if n_full != 8192:
                raise ValueError("CRS bin descrambling assumes 8192 channels, data has "
                                 "{}".format(n_full))
            f_nat = np.arange(n_full)
            f_rev = (f_nat >> 11) | ((f_nat & 0b11111111111) << 2)
            bin_map = (f_rev & 0b1111111111100) | ((f_rev + f_nat) & 0b11)
            self.gain = np.real(g[bin_map, :])[flb:fub, :]
            print("  --> Gains: loaded {}".format(self.gainfile))
            return True
        except (OSError, ValueError) as e:
            print("  --> ERROR: CRS gain file not loaded ({})".format(e))
            return False

    ############################################################
    # PER-FILE PROCESSING
    ############################################################
    def _read_file(self, fd):
        """(channels (t,f,n_channels) complex, UNIX times, sat (t,f,n_ports) or None).
        Gains divided out (if Apply_Gains) and conjugation done so each channel
        matches its name. Offsets are NOT applied here."""
        vis = self._read_vis(fd)
        chans = vis[:, :, self.chan_raw_idx]
        if self.Apply_Gains:
            ga = self.gain[:, self.chan_ports[:, 0]]
            gb = self.gain[:, self.chan_ports[:, 1]]
            chans /= (ga * gb)[np.newaxis, :, :]
        if self.chan_conj.any():
            chans[:, :, self.chan_conj] = np.conj(chans[:, :, self.chan_conj])
        tm = self._read_times(fd)
        sat = None
        if 'sat' in fd and self._layout == 'tfp':
            flb, fub = self.fbounds
            try:
                sat = np.real(np.array(fd['sat'][:, flb:fub, :]))[:, :, self.automap]
            except (ValueError, IndexError):
                sat = None
        return chans, tm, sat

    def _compute_offsets(self, filelist, offset_file_index, offset_time_samples):
        if len(self.cross_idx) == 0:
            print("  --> Offset removal: no cross channels, nothing to do.")
            return
        offset_file = filelist[offset_file_index]
        with h5py.File(os.path.join(self.Data_Directory, offset_file), 'r') as fd:
            chans, _, _ = self._read_file(fd)
        n_use = min(offset_time_samples, chans.shape[0])
        cross = chans[:n_use, :, self.cross_idx]
        self.offset_real = np.median(cross.real, axis=0)    # (freq, n_cross) = V_cross order
        self.offset_imag = np.median(cross.imag, axis=0)
        self._offset = self.offset_real + 1j * self.offset_imag
        self.offset_file = offset_file
        self.offset_n_samples = n_use
        print("  --> Offset removal: median real/imag of the first {} integrations of {} "
              "subtracted from {}; autos untouched.".format(
                  n_use, offset_file, _plural(len(self.cross_idx), 'cross channel')))

    def _check_autos(self):
        """Mapping sanity check: autos must come out real."""
        if len(self.auto_idx) == 0 or len(self.t) == 0:
            return
        sub = self.V[:min(200, len(self.t)), :, self.auto_idx]
        re_max = np.nanmax(np.abs(sub.real), axis=(0, 1))
        im_max = np.nanmax(np.abs(sub.imag), axis=(0, 1))
        for k, r, im in zip(self.auto_idx, re_max, im_max):
            if r > 0 and im / r > 1e-3:
                print("  --> WARNING: auto channel {} '{}' has |imag|/|real| up to {:.2g}; "
                      "autos should be real. Check chmap / instrument.".format(
                          k, self.chnames[int(k)], im / r))

    ############################################################
    # USER-FACING HELPERS
    ############################################################
    @property
    def V_auto(self):
        """Auto channels only, (time, freq, n_auto). Index k here = auto_names[k];
        V index = auto_idx[k]. Built once on first use (a copy of those channels)."""
        if 'V_auto' not in self._cache:
            self._cache['V_auto'] = self.V[:, :, self.auto_idx]
        return self._cache['V_auto']

    @property
    def V_cross(self):
        """Cross channels only, (time, freq, n_cross). Index k here = cross_names[k];
        V index = cross_idx[k]. Built once on first use (a copy of those channels)."""
        if 'V_cross' not in self._cache:
            self._cache['V_cross'] = self.V[:, :, self.cross_idx]
        return self._cache['V_cross']

    def _resolve_port(self, token):
        """keystring ('Ref'), labelled port ('Ref(1)') or port number (1 / '1') -> port."""
        if isinstance(token, (int, np.integer)):
            p = int(token)
        else:
            s = str(token).strip()
            m = re.fullmatch(r'(.*?)\s*\(\s*(\d+)\s*\)', s)
            if m:
                p = int(m.group(2))
                if m.group(1) and p in self.chmap:
                    k = self.keystrings[int(np.where(self.chmap == p)[0][0])]
                    if m.group(1) != k:
                        raise KeyError("Port {} is '{}', not '{}'".format(p, k, m.group(1)))
            elif s.isdigit():
                p = int(s)
            else:
                hits = [int(pp) for pp, k in zip(self.chmap, self.keystrings) if k == s]
                if not hits:
                    raise KeyError("'{}' isn't a keystring. Used ports: {}".format(
                        s, ', '.join(self.port_labels)))
                if len(hits) > 1:
                    raise KeyError("'{}' matches ports {}; use e.g. '{}({})'".format(
                        s, hits, s, hits[0]))
                p = hits[0]
        if p not in self.chmap:
            raise KeyError("Port {} isn't a used port. Used ports: {}".format(
                p, ', '.join(self.port_labels)))
        return p

    def _v_index(self, a, b=None):
        """(V index, conj needed for the requested order)."""
        if b is None:
            if isinstance(a, (int, np.integer)):
                idx = int(a)
                if not -self.n_channels <= idx < self.n_channels:
                    raise IndexError("V channel {} out of range (0-{})".format(
                        idx, self.n_channels - 1))
                return idx % self.n_channels, False
            parts = _X_SPLIT.split(str(a).strip())
            if len(parts) != 2:
                if len(parts) == 1:          # one port -> its auto
                    a = b = parts[0]
                else:
                    raise KeyError("Give a channel as 'A x B', two ports, or an index; "
                                   "got {!r}".format(a))
            else:
                a, b = parts
        pa, pb = self._resolve_port(a), self._resolve_port(b)
        for k, (x, y) in enumerate(self.chan_ports):
            if x == pa and y == pb:
                return k, False
            if x == pb and y == pa:
                return k, True
        raise KeyError("No channel for ports {} x {}".format(pa, pb))

    def chan_index(self, a, b=None, array='V', return_conj=False):
        """Index of a channel in V (default), 'V_auto' or 'V_cross'.
            chan_index(4)                          V index 4 (checked)
            chan_index('Ref(1) x Dish(3)')         by name
            chan_index('Ref', 'Dish')              keystrings / 'Ref(1)' / port numbers
            chan_index('Ref')                      the Ref auto
            chan_index('Ref', 'Dish', array='V_cross')   its index in V_cross
        A request for B x A of a channel stored as A x B returns that channel;
        with return_conj=True you also get True = your order is its conjugate."""
        k, c = self._v_index(a, b)
        array = str(array)
        if array == 'V_auto':
            if not self.chan_is_auto[k]:
                raise KeyError("'{}' is a cross, not in V_auto".format(self.chnames[k]))
            k = int(self.chan_sub_idx[k])
        elif array == 'V_cross':
            if self.chan_is_auto[k]:
                raise KeyError("'{}' is an auto, not in V_cross".format(self.chnames[k]))
            k = int(self.chan_sub_idx[k])
        elif array != 'V':
            raise ValueError("array must be 'V', 'V_auto' or 'V_cross'")
        return (k, c) if return_conj else k

    def auto_index(self, a):
        """Index in V_auto of a port's auto."""
        return self.chan_index(a, array='V_auto')

    def cross_index(self, a, b=None):
        """Index in V_cross of a cross channel."""
        return self.chan_index(a, b, array='V_cross')

    def get_chan(self, a, b=None):
        """Complex data (time, freq) of one channel in exactly the order asked."""
        k, c = self._v_index(a, b)
        return np.conj(self.V[:, :, k]) if c else self.V[:, :, k]

    def magnitude(self, a, b=None):
        """np.abs() of one channel (time, freq) - right for autos and crosses."""
        return np.abs(self.V[:, :, self._v_index(a, b)[0]])

    def is_auto(self, a, b=None):
        return bool(self.chan_is_auto[self._v_index(a, b)[0]])

    def print_channels(self):
        """Table of every channel: V index, V_auto / V_cross index, name, ports, ..."""
        tab = self.channel_table
        show_dish = any(d is not None for d in tab['dish_a'])
        hdr = ['V', 'V_auto', 'V_cross', 'name', 'type', 'ports'] + (['dishes'] if show_dish else [])
        rows = []
        for k, r in tab.iterrows():
            row = [str(k),
                   '' if pandas.isna(r['V_auto']) else str(r['V_auto']),
                   '' if pandas.isna(r['V_cross']) else str(r['V_cross']),
                   r['name'], r['type'], '{} x {}'.format(r['port_a'], r['port_b'])]
            if show_dish:
                row.append(str(r['dish_a']) if r['dish_a'] == r['dish_b']
                           else '{} x {}'.format(r['dish_a'], r['dish_b']))
            rows.append(row)
        w = [max(len(h), *(len(r[c]) for r in rows)) for c, h in enumerate(hdr)]
        line = '  '.join('{:<%d}' % x for x in w)
        print('{} channels from the {} ({}, {}). Use np.abs() for magnitudes:'
              .format(self.n_channels, INSTRUMENT_LONGNAME[self.instrument],
                      _plural(len(self.auto_idx), 'auto'), _plural(len(self.cross_idx), 'cross')))
        print('  ' + line.format(*hdr))
        print('  ' + '  '.join('-' * x for x in w))
        for r in rows:
            print('  ' + line.format(*r))