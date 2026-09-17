# _____ _________________
#/  __ \  _  | ___ \ ___ \
#| /  \/ | | | |_/ / |_/ /  _ __  _   _
#| |   | | | |    /|    /  | '_ \| | | |
#| \__/\ \_/ / |\ \| |\ \ _| |_) | |_| |
# \____/\___/\_| \_\_| \_(_) .__/ \__, |
#                          | |     __/ |
#                          |_|    |___/
## 08/21 Unified auto/cross channel model (both _initialize_corr and
##   _initialize_crs): removed crossmap/V_cross and the separate "auto
##   channel == V array" assumption. Every saved product between two used
##   ports is now kept as one flat, non-repeating list: self.chan_prod
##   (physical port pairs), self.chan_is_auto, and self.V (last axis =
##   self.n_channels, complex-native -- take np.abs() downstream, never
##   .real). New auto_only=False constructor arg restricts to autos only
##   when set True. chmap/automap/gain/sat stay port-indexed (n_ports =
##   len(site_class.chmap), unaffected by this change) -- self.n_dishes is
##   now len(chmap) rather than an assumed n_channels/2 pairing.
##   remove_real_imag_offset's offset_crossmap keeps meaning "list of raw
##   product indices" (positions in self.prod) exactly as before, and the
##   subtraction now happens directly on `vis` in raw-product-index space
##   (before channel selection), so it applies correctly regardless of
##   whether that product ends up kept as an auto or cross channel.
##   Also fixed a bug in _initialize_crs's main loop: it was writing
##   vis[...,automap].real into self.V once via automap AND AGAIN (for
##   autos only) via the old crossmap-population step below it -- the first,
##   redundant assignment is gone; V is populated once, from self.keep_idx,
##   uniformly for every channel. The two large commented-out draft __init__
##   blocks previously at the end of this file (dead code, never executed)
##   were dropped for clarity -- shout if you wanted to keep any of that
##   reference/history.
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
class Corr_Data:
    def __init__(self,Data_Directory,Gain_Directory,site_class,
                 Data_File_Index=[0,-1],Load_Gains=True,Fix_Gains=False,
                 Apply_Gains=True,Gain_Params=[1.0,24.0],
                 fbounds=[0,1024],use_ctime=False,
                 esystem=False,

                 crs=False,
                 auto_only=False,
                 # NEW OFFSET PARAMETERS
                 remove_real_imag_offset=False,
                 offset_crossmap=[],
                 offset_file_index=0,
                 offset_time_samples=400,
                 offset_freq_index=0,
                 plot_offset_diagnostic=True):
        """
        auto_only : bool, default False
            Every saved product is already a fully-formed correlation --
            auto (real, imag~0) or cross (complex, imag!=0) -- there is no
            "raw"/uncorrelated retrieval mode. By default every saved
            auto+cross product between used channels is kept as one flat,
            non-repeating list (self.chan_prod / self.chan_is_auto /
            self.V, last axis length self.n_channels). Pass auto_only=True
            to restrict to only the auto-correlations (channel count then
            equals the number of used physical ports).
        """
        if crs:
            self._initialize_crs(
                Data_Directory,
                Gain_Directory,
                site_class,
                Data_File_Index,
                Load_Gains,
                Fix_Gains,
                Apply_Gains,
                Gain_Params,
                fbounds,
                auto_only,
                esystem,
                remove_real_imag_offset,
                offset_crossmap,
                offset_file_index,
                offset_time_samples,
                offset_freq_index,
                plot_offset_diagnostic,
            )
        else:
            self._initialize_corr(
                Data_Directory,
                Gain_Directory,
                site_class,
                Data_File_Index,
                Load_Gains,
                Fix_Gains,
                Apply_Gains,
                Gain_Params,
                fbounds,
                use_ctime,
                auto_only,
                esystem,
                remove_real_imag_offset,
                offset_crossmap,
                offset_file_index,
                offset_time_samples,
                offset_freq_index,
                plot_offset_diagnostic,
            )

    def _initialize_corr(
                self,
                Data_Directory,
                Gain_Directory,
                site_class,
                Data_File_Index,
                Load_Gains,
                Fix_Gains,
                Apply_Gains,
                Gain_Params,
                fbounds,
                use_ctime,
                auto_only,
                esystem,
                remove_real_imag_offset,
                offset_crossmap,
                offset_file_index,
                offset_time_samples,
                offset_freq_index,
                plot_offset_diagnostic,
        ):
            self.Data_Directory=Data_Directory
            self.Gain_Directory=Gain_Directory
            filelist=np.sort([x for x in os.listdir(self.Data_Directory) if ".lock" not in x])[:-1]
            self.filenames=filelist[Data_File_Index[0]:Data_File_Index[1]]
            n_files=len(self.filenames)
            print('Initializing Correlator Class using:')
            print("  --> "+self.Data_Directory)
            fd=h5py.File(self.Data_Directory+self.filenames[0],'r')
            self.fbounds=fbounds
            flb=fbounds[0]
            fub=fbounds[1]
            vis=fd['vis'][:,flb:fub,:]
            if 'CHIME' in site_class.name:
                if esystem==True:
                    Cevals=np.tile(np.array(fd['eval']).transpose(2,0,1)[:,:,:,np.newaxis][:,:,0,:],fd['evec'].shape[2])
                    Cevecs=np.abs(np.array(fd['evec'][:,0,:,:]).transpose(2,0,1))**2.0
                    vis=Cevals*Cevecs
                else:
                    vis=np.array(fd['vis']).transpose(2,0,1)[:,flb:fub,:]
            if use_ctime==False:
                self.t0=1e-9*fd['index_map']['time']['irigb_time'][0]
            if use_ctime==True:
                self.t0=fd['index_map']['time']['ctime'][0]
                corrtoff=(2.56e-6)*fd['index_map']['time']['fpga_count'][0]
            self.freq=np.array([i[0] for i in fd['index_map']['freq'][flb:fub]])

            self.prod=fd['index_map']['prod'][:]
            prodmat=np.array([element for tupl in self.prod for element in tupl]).reshape(len(self.prod),2)
            ## Unified auto+cross channel model: n_ports is the number of
            ## physical ports actually used (from site_class.chmap); chmap/
            ## automap/gain/sat stay port-indexed (n_ports long). chan_prod/
            ## chan_is_auto/self.V are one flat, non-repeating list covering
            ## every saved product between two used ports (auto included by
            ## default -- pass auto_only=True in Corr_Data.__init__ to
            ## restrict to autos only):
            n_ports=min(len(site_class.chmap),int(prodmat[:,0].max()+1))
            self.chmap=np.array(site_class.chmap[:n_ports]).astype(int)
            used_ports=set(self.chmap.tolist())
            self.automap=np.zeros(n_ports).astype(int)
            for i,j in enumerate(self.chmap):
                self.automap[i]=np.intersect1d(np.where(prodmat[:,0]==j),np.where(prodmat[:,1]==j))[0]
            keep_idx=[]
            for k,(a,b) in enumerate(prodmat):
                if a in used_ports and b in used_ports:
                    if auto_only and a!=b:
                        continue
                    keep_idx.append(k)
            keep_idx=np.array(keep_idx,dtype=int)
            self.keep_idx=keep_idx
            self.n_channels=len(keep_idx)
            self.n_dishes=len(self.chmap)
            self.chan_prod=prodmat[keep_idx]
            self.chan_is_auto=self.chan_prod[:,0]==self.chan_prod[:,1]
            self.V=np.zeros((n_files,vis.shape[0],vis.shape[1],self.n_channels)).astype(complex)
            self.t=np.zeros((n_files,vis.shape[0]))
            self.sat=np.zeros((n_files,vis.shape[0],vis.shape[1],n_ports))
            self.gainfile=os.listdir(self.Gain_Directory)[0]
            digital_gain=np.ones((len(self.freq),n_ports))
            self.gain_coeffs=np.ones(len(self.freq))
            self.gain_exp=np.ones(vis.shape[2])
            if Load_Gains==True:
                try:
                    fg=h5py.File(self.Gain_Directory+self.gainfile,'r')
                    self.gain_coeffs=fg['gain_coeff'][0]
                    self.gain_exp=fg['gain_exp'][0]
                    digital_gain=fg['gain_coeff'][0]
                    digital_gain*=np.power(2,fg['gain_exp'][0])[np.newaxis,:]
                    fg.close()
                except OSError:
                    print("  --> ERROR: Gain file not found")
            if Fix_Gains==True:
                digital_gain=Gain_Params[0]*np.array((2**Gain_Params[1])*np.ones((vis.shape[1],n_ports))).astype(complex)
                self.gain_coeffs=Gain_Params[0]*np.ones(len(self.freq))
                self.gain_exp=Gain_Params[1]*np.ones(vis.shape[2])
            self.gain=digital_gain.real[flb:fub,:]
            fd.close()
            ############################################################
            # OFFSET ESTIMATION (NEW BLOCK — FROM YOUR SCRIPT)
            ############################################################
            real_offset=None
            imag_offset=None
            if remove_real_imag_offset and len(offset_crossmap)>0:
                print("\nComputing offsets using reference file:",offset_file_index)
                offset_file=filelist[offset_file_index]
                fd_offset=h5py.File(self.Data_Directory+offset_file,'r')
                vis_offset=fd_offset['vis'][:,flb:fub,:]
                n_offset_time=min(offset_time_samples,vis_offset.shape[0])
                vis_subset=vis_offset[:n_offset_time,:,offset_crossmap]
                if Apply_Gains:
                    for ci,k in enumerate(offset_crossmap):
                        prod_pair=self.prod[k]
                        g1=self.gain[:,prod_pair[0]]
                        g2=self.gain[:,prod_pair[1]]
                        vis_subset[:,:,ci]/=(g1*g2)[np.newaxis,:]
                real_offset=np.median(vis_subset.real,axis=0)
                imag_offset=np.median(vis_subset.imag,axis=0)
                fd_offset.close()
            ############################################################
            # MAIN FILE LOOP (ORIGINAL STRUCTURE)
            ############################################################
            print("Assigning array values by reading in data files:")
            for i,file in enumerate(self.filenames):
                try:
                    print("\r  --> Loading File: {}/{}".format(self.filenames[i],self.filenames[-1]),end="")
                    fd_n=h5py.File(self.Data_Directory+file,'r')
                    vis=fd_n['vis'][:,flb:fub,:]
                    if 'CHIME' in site_class.name:
                        if esystem==True:
                            Cevals=np.tile(np.array(fd_n['eval']).transpose(2,0,1)[:,:,:,np.newaxis][:,:,0,:],fd_n['evec'].shape[2])
                            Cevecs=np.abs(np.array(fd_n['evec'][:,0,:,:]).transpose(2,0,1))**2.0
                            vis=Cevals*Cevecs
                        else:
                            vis=np.array(fd_n['vis']).transpose(2,0,1)[:,flb:fub,:]
                    tm=(2.56e-6)*np.array(fd_n['index_map']['time']['fpga_count'])-corrtoff
                    if use_ctime==False:
                        tm=np.array(fd_n['index_map']['time']['irigb_time'])
                    prod=fd_n['index_map']['prod'][:]
                    ################################################
                    # APPLY GAINS
                    ################################################
                    if Apply_Gains:
                        for ii,pp in enumerate(prod):
                            vis[:,:,ii]/=(self.gain[:,pp[0]]*self.gain[:,pp[1]])[np.newaxis,:]
                    ################################################
                    # OPTIONAL REAL/IMAG OFFSET REMOVAL -- raw product index
                    # space, BEFORE channel selection below, so it applies
                    # uniformly regardless of whether that raw product ends
                    # up kept as an auto or cross channel:
                    ################################################
                    if remove_real_imag_offset:
                        for oi,k in enumerate(offset_crossmap):
                            vis[:,:,k]=(vis[:,:,k].real-real_offset[:,oi][np.newaxis,:]) \
                                      +1j*(vis[:,:,k].imag-imag_offset[:,oi][np.newaxis,:])
                    ################################################
                    # POPULATE SATURATION (port-indexed, unchanged):
                    ################################################
                    for j,k in enumerate(self.automap):
                        try:
                            self.sat[i,:,:,j]=fd_n['sat'][:,flb:fub,k].real
                        except KeyError:
                            pass
                    ################################################
                    # POPULATE V: one flat, non-repeating list of every kept
                    # auto+cross product (self.keep_idx / self.chan_prod),
                    # complex-native -- downstream code takes np.abs():
                    ################################################
                    for j,k in enumerate(self.keep_idx):
                        self.V[i,:,:,j]=vis[:,:,k]
                    self.t[i,:]=tm
                    fd_n.close()
                except OSError:
                    print('\nSkipping file: {}'.format(file))
            print("\n  --> Finished. Reshaping arrays.")
            ############################################################
            # RESHAPING
            ############################################################
            self.V=self.V.reshape((n_files*vis.shape[0],vis.shape[1],self.n_channels))
            self.t=self.t.reshape(n_files*vis.shape[0])
            self.sat=self.sat.reshape((n_files*vis.shape[0],vis.shape[1],self.sat.shape[3]))
            timedeltas=np.array([datetime.timedelta(seconds=x) for x in self.t])
            dt0=datetime.datetime.fromtimestamp(self.t0,pytz.timezone('America/Montreal')).astimezone(pytz.utc)
            self.t_arr_datetime=dt0+timedeltas
            if 'CHIME' in site_class.name:
                self.t_arr_datetime=np.array([datetime.datetime.fromtimestamp(x,pytz.utc) for x in self.t])
            if 'D3A' in site_class.name:
                self.t_arr_datetime=np.array([datetime.datetime.fromtimestamp(1e-9*x,pytz.utc) for x in self.t])
            self.t_index=np.arange(len(self.t))


    def _initialize_crs(
                self,
                Data_Directory,
                Gain_Directory,
                site_class,
                Data_File_Index,
                Load_Gains,
                Fix_Gains,
                Apply_Gains,
                Gain_Params,
                fbounds,
                auto_only,
                esystem,
                remove_real_imag_offset,
                offset_crossmap,
                offset_file_index,
                offset_time_samples,
                offset_freq_index,
                plot_offset_diagnostic,
        ):
            self.Data_Directory=Data_Directory
            self.Gain_Directory=Gain_Directory

            filelist = sorted(
            [x for x in os.listdir(self.Data_Directory) if ".lock" not in x],
            key=lambda x: int(x.split(".")[0]))

            self.filenames=filelist[Data_File_Index[0]:Data_File_Index[1]]
            n_files=len(self.filenames)
            print('Initializing Correlator Class using:')
            print("  --> "+self.Data_Directory)
            fd=h5py.File(self.Data_Directory+self.filenames[0],'r')
            self.fbounds=fbounds
            flb=fbounds[0]
            fub=fbounds[1]
            ############################################################
            # READ FIRST FILE / INITIALIZE ARRAYS
            ############################################################
            # CRS visibilities are already stored as:
            # (time, frequency, product)
            vis = fd['vis'][:, flb:fub, :]
            # Absolute UNIX timestamp of first integration
            self.t0 = fd['index_map']['time'][0]
            # Full (unbounded) frequency axis length, needed below to figure
            # out whether a gain file covers the whole band or was already
            # pre-truncated to [flb:fub]:
            n_freq_full = fd['index_map']['freq'].shape[0]
            # Frequency axis (already 1D for CRS)
            self.freq = np.array(fd['index_map']['freq'][flb:fub])
            ############################################################
            # CRS PRODUCT MAP
            #
            # The CRS board currently writes all 36 visibility products,
            # but the HDF5 index_map/prod dataset is not populated
            # correctly, so we define the ordering explicitly.
            ############################################################
            self.prod = np.array([
                [0,0],
                [0,1],
                [0,2],
                [0,3],
                [1,1],
                [1,2],
                [1,3],
                [1,4],
                [2,2],
                [2,3],
                [2,4],
                [2,5],
                [3,3],
                [3,4],
                [3,5],
                [3,6],
                [4,4],
                [4,5],
                [4,6],
                [4,7],
                [5,5],
                [5,6],
                [5,7],
                [0,4],
                [6,6],
                [6,7],
                [0,5],
                [1,5],
                [7,7],
                [0,6],
                [1,6],
                [2,6],
                [0,7],
                [1,7],
                [2,7],
                [3,7]
            ], dtype=int)

            if vis.shape[2] != len(self.prod):
                raise RuntimeError(
                    f"Expected 36 CRS products, but vis has {vis.shape[2]}"
                )
            ############################################################
            # CHANNELS USED IN THIS OBSERVATION
            #
            # Unified auto+cross channel model: n_ports is the number of
            # physical CRS ports actually used (from site_class.chmap);
            # chmap/automap/sat stay port-indexed (n_ports long).
            # chan_prod/chan_is_auto/self.V are one flat, non-repeating list
            # covering every one of the 36 saved products between two used
            # ports (auto included by default -- pass auto_only=True in
            # Corr_Data.__init__ to restrict to autos only).
            ############################################################
            n_ports = len(site_class.chmap)
            self.chmap = np.array(site_class.chmap[:n_ports]).astype(int)
            used_ports = set(self.chmap.tolist())
            ############################################################
            # FIND AUTO-CORRELATION PRODUCT FOR EACH PORT (sat lookup only)
            ############################################################
            self.automap = np.zeros(n_ports, dtype=int)
            for i, ch in enumerate(self.chmap):
                matches = np.where(
                    (self.prod[:,0] == ch) &
                    (self.prod[:,1] == ch)
                )[0]
                if len(matches) == 0:
                    raise RuntimeError(
                        f"No auto-correlation found for channel {ch}"
                    )
                self.automap[i] = matches[0]
            ############################################################
            # FLAT AUTO+CROSS CHANNEL LIST
            ############################################################
            keep_idx = []
            for k, (a, b) in enumerate(self.prod):
                if a in used_ports and b in used_ports:
                    if auto_only and a != b:
                        continue
                    keep_idx.append(k)
            keep_idx = np.array(keep_idx, dtype=int)
            self.keep_idx = keep_idx
            self.n_channels = len(keep_idx)
            self.n_dishes = n_ports
            self.chan_prod = self.prod[keep_idx]
            self.chan_is_auto = self.chan_prod[:,0] == self.chan_prod[:,1]
            ############################################################
            # PRE-ALLOCATE OUTPUT ARRAYS
            ############################################################
            n_time = vis.shape[0]
            n_freq = vis.shape[1]
            self.V = np.zeros(
                (n_files, n_time, n_freq, self.n_channels),
                dtype=complex
            )
            self.t = np.zeros(
                (n_files, n_time)
            )
            self.sat = np.zeros(
                (n_files, n_time, n_freq, n_ports)
            )
            # CRS convention (confirmed from real data): Gain_Directory is
            # the path to one specific per-acquisition .npy file, already
            # chosen by the caller (e.g. gaindir = gdir + '<file>.npy' in
            # the analysis notebook) -- NOT a folder to search, since CRS
            # gains are one-per-acquisition rather than one-per-channel.
            self.gainfile = os.path.basename(self.Gain_Directory)
            ############################################################
            # LOAD CRS DIGITAL GAINS (.npy)
            #
            # Unlike the iceboard's gain_coeff/gain_exp h5 pair, CRS digital
            # gains are a single plain real-valued .npy array, shape
            # (n_channels_hw, n_freq_full) -- confirmed 8x8192 in the field.
            #
            # IMPORTANT: n_channels_hw (the number of physical CRS board
            # inputs, i.e. the range of indices used in self.prod above) is
            # NOT the same thing as n_ports (len(site_class.chmap), the
            # subset of physical ports THIS analysis cares about -- e.g. 4 of
            # the 8 physical inputs for a given flight) -- and it's not
            # self.n_channels either, which is now a *product* count (every
            # kept auto+cross channel), not a port count at all. self.gain
            # is indexed below by raw self.prod entries (0..n_channels_hw-1),
            # so it must always be sized to all physical inputs, regardless
            # of how many of them site_class.chmap actually uses.
            #
            # CRS gain files also store their frequency axis in the FPGA's
            # native bit-reversed FFT bin order rather than natural
            # ascending frequency order (confirmed 08/2026 -- this is what
            # made early gain-vs-frequency plots look like scrambled noise
            # instead of a smooth bandpass). bin_map below undoes that
            # permutation. It's only valid across the complete 8192-channel
            # band, so it must be applied BEFORE any [flb:fub] slicing.
            ############################################################
            n_channels_hw = int(self.prod.max()) + 1  # physical CRS inputs (8)
            self.gain = np.ones((len(self.freq), n_channels_hw))
            self.gain_coeffs = np.ones(len(self.freq))
            self.gain_exp = np.ones(vis.shape[2])
            if Load_Gains:
                try:
                    if os.path.isdir(self.Gain_Directory):
                        raise ValueError(
                            "Gain_Directory is a directory ({}); CRS gain "
                            "files are one-per-acquisition, not one-per-"
                            "channel -- point Gain_Directory at the exact "
                            ".npy file for this acquisition instead (e.g. "
                            "gaindir = gdir + '<specific_file>.npy').".format(
                                self.Gain_Directory
                            )
                        )
                    digital_gain = np.load(self.Gain_Directory, allow_pickle=True)
                    if digital_gain.shape == (n_channels_hw, n_freq_full):
                        digital_gain = digital_gain.T
                    elif digital_gain.shape != (n_freq_full, n_channels_hw):
                        raise ValueError(
                            "CRS gain array has shape {}, expected ({}, {}) "
                            "or ({}, {}).".format(
                                digital_gain.shape,
                                n_channels_hw, n_freq_full,
                                n_freq_full, n_channels_hw,
                            )
                        )
                    if n_freq_full != 8192:
                        raise ValueError(
                            "CRS bin descrambling below is hardcoded for an "
                            "8192-channel band, but n_freq_full={}. Refusing "
                            "to silently apply a mismatched permutation.".format(
                                n_freq_full
                            )
                        )
                    # Undo the FPGA's bit-reversed frequency bin ordering:
                    f_nat = np.arange(n_freq_full)
                    f_rev = (f_nat >> 11) | ((f_nat & 0b11111111111) << 2)
                    bin_map = (f_rev & 0b1111111111100) | ((f_rev + f_nat) & 0b11)
                    digital_gain = digital_gain[bin_map, :]
                    self.gain = digital_gain.real[flb:fub, :]
                except (OSError, ValueError) as e:
                    print("  --> ERROR: CRS gain file not loaded:", e)
                    print("  --> Falling back to unity gains (Apply_Gains will be a no-op).")
            if Fix_Gains:
                digital_gain = Gain_Params[0]*np.array(
                    (2**Gain_Params[1])*np.ones((n_freq, n_channels_hw))
                ).astype(complex)
                self.gain_coeffs = Gain_Params[0]*np.ones(len(self.freq))
                self.gain_exp = Gain_Params[1]*np.ones(vis.shape[2])
                self.gain = digital_gain.real
            fd.close()
            ############################################################
            # OFFSET ESTIMATION (NEW BLOCK)
            ############################################################
            real_offset = None
            imag_offset = None
            if remove_real_imag_offset and len(offset_crossmap) > 0:
                print("\nComputing offsets using reference file:", offset_file_index)
                offset_file = filelist[offset_file_index]
                fd_offset = h5py.File(self.Data_Directory + offset_file, 'r')
                # Read visibilities from reference file
                vis_offset = fd_offset['vis'][:, flb:fub, :]
                # Number of integrations to use for offset estimation
                n_offset_time = min(offset_time_samples, vis_offset.shape[0])
                # Select only requested cross-correlations
                vis_subset = vis_offset[:n_offset_time, :, offset_crossmap].astype(np.complex128)
                # Optionally apply gains before estimating offsets
                if Apply_Gains:
                    for ci, k in enumerate(offset_crossmap):
                        prod_pair = self.prod[k]
                        g1 = self.gain[:, prod_pair[0]]
                        g2 = self.gain[:, prod_pair[1]]
                        vis_subset[:, :, ci] /= (g1 * g2)[np.newaxis, :]
                # Median real/imaginary offset for every frequency and baseline
                real_offset = np.median(vis_subset.real, axis=0)
                imag_offset = np.median(vis_subset.imag, axis=0)
                print("Computed complex offsets using {} integrations.".format(n_offset_time))
                fd_offset.close()
             ############################################################
            # MAIN FILE LOOP (CRS STRUCTURE)
            ############################################################
            print("Assigning array values by reading in data files:")
            for i, file in enumerate(self.filenames):
                try:
                    print("\r  --> Loading File {}/{}".format(i+1, n_files), end="")
                    fd_n = h5py.File(self.Data_Directory + file, 'r')
                    if fd_n['vis'].shape[0] == 0:
                        print("\nSkipping incomplete file:", file)
                        fd_n.close()
                        continue
                    vis = fd_n['vis'][:, flb:fub, :]
                    ################################################
                    # APPLY GAINS (mirrors _initialize_corr; previously
                    # missing entirely from the CRS branch, so gains were
                    # loaded/offset-corrected-for but never actually
                    # divided out of the visibilities)
                    ################################################
                    if Apply_Gains:
                        for ii, pp in enumerate(self.prod):
                            vis[:,:,ii] /= (self.gain[:,pp[0]]*self.gain[:,pp[1]])[np.newaxis,:]
                    ################################################
                    # OPTIONAL REAL/IMAG OFFSET REMOVAL -- raw product index
                    # space, BEFORE channel selection below, so it applies
                    # uniformly regardless of whether that raw product ends
                    # up kept as an auto or cross channel:
                    ################################################
                    if remove_real_imag_offset:
                        for oi,k in enumerate(offset_crossmap):
                            vis[:,:,k]=(vis[:,:,k].real-real_offset[:,oi][np.newaxis,:]) \
                                      +1j*(vis[:,:,k].imag-imag_offset[:,oi][np.newaxis,:])
                    ################################################
                    # CRS TIME ARRAY
                    ################################################
                    tm = np.array(fd_n['index_map']['time'])
                    ################################################
                    # POPULATE SATURATION (port-indexed, unchanged):
                    ################################################
                    for j, k in enumerate(self.automap):
                        try:
                            sat_data = fd_n['sat'][:, flb:fub, k].real
                            if sat_data.shape == self.sat[i, :, :, j].shape:
                                self.sat[i, :, :, j] = sat_data
                            else:
                                print("\nSkipping sat for incomplete file:", file)
                        except KeyError:
                            pass
                    ################################################
                    # POPULATE V: one flat, non-repeating list of every kept
                    # auto+cross product (self.keep_idx / self.chan_prod),
                    # complex-native -- downstream code takes np.abs():
                    ################################################
                    for j,k in enumerate(self.keep_idx):
                        self.V[i,:,:,j]=vis[:,:,k]
                    self.t[i,:]=tm
                    fd_n.close()
                except OSError:
                    print('\nSkipping file: {}'.format(file))
            print("\n  --> Finished. Reshaping arrays.")
            ############################################################
            # RESHAPE ARRAYS
            ############################################################
            self.V = self.V.reshape(
                (n_files * n_time, n_freq, self.n_channels)
            )
            self.t = self.t.reshape(
                n_files * n_time
            )
            self.sat = self.sat.reshape(
                (n_files * n_time, n_freq, self.sat.shape[3])
            )
            ############################################################
            # CRS TIMESTAMPS
            #
            # CRS time is already UNIX seconds
            ############################################################
            self.t_arr_datetime = np.array(
                [
                    datetime.datetime.fromtimestamp(x, pytz.utc)
                    for x in self.t
                ]
            )
            self.t_index = np.arange(len(self.t))