#  _____ _____ _____ _____ _____ ________   __
# /  ___|_   _|_   _|  ___/  ___|| ___ \ \ / /
# \ `--.  | |   | | | |__ \ `--. | |_/ /\ V /
#  `--. \ | |   | | |  __| `--. \|  __/  \ /
# /\__/ /_| |_  | | | |___/\__/ /| |     | |
# \____/ \___/  \_/ \____/\____(_)_|     \_/
#
## Site-specific information (what was plugged in where, where the dishes are,
## how they point, how the feeds are rotated), stored in an NPZ file and read
## back into a Site object.
##
##   site_utils.site_template()                 # print a fill-in-the-blanks notebook cell
##   site_utils.make_site_file(...)             # create + validate + save a site file
##   site_utils.show_site_file('X_config.npz')  # pretty-print any site file (old or new)
##   site_utils.edit_site_file('X_config.npz', polarizations={3: 'NE'})   # change one thing
##   site = site_utils.Site('X_config.npz')     # load (what corr.py uses)
##
## FILE FORMAT v2 (written by make_site_file). Conventions:
##   * chmap[i]            correlator input (port) that carried channel i
##   * keystrings[i]       your name for channel i
##   * chmap_to_dishes[i]  which dish/instrument (0..n_dishes-1) channel i belongs to
##   * dish coords are East, North, Up in metres from `origin` [lat, lon, elev]
##     if an origin is given; otherwise each dish is given as [lat, lon, elev].
##     coords_mode can mix the two per dish ('m' / 'latlon'). Every dish is
##     stored in both forms. NaN row = location irrelevant (reference board etc.)
##   * pointing per dish = [tilt from zenith, azimuth of the tilt], degrees,
##     azimuth measured from North towards East. Zenith = [0, 0].
##   * polarization per channel = feed angle in degrees from North towards
##     East (compass names allowed: 'N', 'NE', 'NW', 'EW', ...). Polarization is
##     an axis, so NE == SW, NW == SE, N == S. None/NaN = not applicable.
##
## Backward compatibility: Site still has .name, .n_dishes, .n_channels,
## .chmap, .keystrings, .origin, and the old PER-CHANNEL arrays .coords
## (East/North/Up m), .pointings and .polarizations (unit vectors in the same
## East/North/Up frame), with the same meaning as the old files. The per-dish
## information lives in .dish_coords, .dish_coords_geo, .dish_pointing,
## .dish_names, .chmap_to_dishes, and per-channel .pol_angle / .pol_labels.
## Old-format files still load; the dish grouping is inferred from channels
## sharing identical coordinates.

import os
import re
import shutil
import numpy as np
try:
    import pygeodesy  # kept for older code that expected it here; not required
except ImportError:
    pygeodesy = None

FORMAT_VERSION = 2
N_PORTS = {'crs': 8, 'ice': 16}
_MODE_ALIASES = {
    'enu': 'enu', 'm': 'enu', 'meters': 'enu', 'metres': 'enu', 'relative': 'enu',
    'geodetic': 'geodetic', 'latlon': 'geodetic', 'lat/lon': 'geodetic', 'geo': 'geodetic',
    'lla': 'geodetic',
}

############################################################
# GEODESY (WGS84, no external dependency)
############################################################
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def _geo_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.array([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - _E2) + h) * np.sin(lat)])


def _ecef_to_geo(xyz):
    x, y, z = xyz
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(10):
        N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - _E2 * N / (N + h)))
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    return np.array([np.degrees(lat), np.degrees(lon), h])


def _enu_matrix(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def geo_to_enu(geo, origin):
    """[lat, lon, elev] -> [East, North, Up] metres relative to origin [lat, lon, elev]."""
    geo = np.asarray(geo, float)
    if np.any(np.isnan(geo)) or np.any(np.isnan(origin)):
        return np.full(3, np.nan)
    return _enu_matrix(origin[0], origin[1]) @ (_geo_to_ecef(*geo) - _geo_to_ecef(*origin))


def enu_to_geo(enu, origin):
    """[East, North, Up] metres relative to origin -> [lat, lon, elev]."""
    enu = np.asarray(enu, float)
    if np.any(np.isnan(enu)) or np.any(np.isnan(origin)):
        return np.full(3, np.nan)
    return _ecef_to_geo(_geo_to_ecef(*origin) + _enu_matrix(origin[0], origin[1]).T @ enu)


############################################################
# ANGLES
############################################################
_COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
_COMPASS_DEG = {k: 22.5 * i for i, k in enumerate(_COMPASS)}
_COMPASS_DEG.update({'NS': 0.0, 'SN': 0.0, 'EW': 90.0, 'WE': 90.0})
_NONE_WORDS = ('', 'none', 'nan', 'na', 'n/a', '-')


def _pol_to_deg(p):
    """Polarization entry -> (degrees from North towards East, label)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return np.nan, 'None'
    if isinstance(p, (int, float, np.integer, np.floating)):
        return float(p) % 360.0, '{:g}'.format(float(p) % 360.0)
    s = str(p).strip()
    if s.lower() in _NONE_WORDS:
        return np.nan, 'None'
    if s.upper() in _COMPASS_DEG:
        return _COMPASS_DEG[s.upper()], s.upper()
    try:
        v = float(s.rstrip('°').replace('deg', ''))
        return v % 360.0, '{:g}'.format(v % 360.0)
    except ValueError:
        raise ValueError(
            "Polarization {!r} not understood. Use degrees from North towards East "
            "(e.g. 45), a compass direction (N, NE, E, SE, ..., NW, or 'NS'/'EW'), "
            "or None for no polarization. ('X'/'Y' are ambiguous - give the actual "
            "direction.)".format(p))


def _pointing_entry(p):
    """One dish's pointing -> [tilt_deg, az_deg]."""
    if p is None:
        return [0.0, 0.0]
    if isinstance(p, str):
        if p.strip().lower() == 'zenith':
            return [0.0, 0.0]
        raise ValueError("Pointing {!r} not understood; use 'zenith' or [tilt, azimuth] "
                         "in degrees.".format(p))
    p = np.asarray(p, float).ravel()
    if p.shape != (2,):
        raise ValueError("Pointing {!r} should be 'zenith' or [tilt from zenith, azimuth] "
                         "in degrees.".format(p))
    if not 0 <= p[0] <= 90:
        raise ValueError("Pointing tilt {} deg is outside 0-90 deg from zenith.".format(p[0]))
    return [float(p[0]), float(p[1]) % 360.0]


def pointing_vector(tilt_deg, az_deg):
    """Unit boresight vector in East/North/Up."""
    t, a = np.radians(tilt_deg), np.radians(az_deg)
    return np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])


def pol_vector(pol_deg, tilt_deg=0.0, az_deg=0.0):
    """Unit feed-polarization vector in East/North/Up: the horizontal direction
    pol_deg (from North towards East), projected perpendicular to the boresight."""
    if np.isnan(pol_deg):
        return np.full(3, np.nan)
    a = np.radians(pol_deg)
    v = np.array([np.sin(a), np.cos(a), 0.0])
    b = pointing_vector(tilt_deg, az_deg)
    v = v - np.dot(v, b) * b
    return v / np.linalg.norm(v)


############################################################
# SITE CLASS
############################################################
class Site:
    def __init__(self, NPZ_Filename):
        self.filename = NPZ_Filename
        with np.load(NPZ_Filename) as cfg:
            d = {k: cfg[k] for k in cfg.files}
        self.name = str(d['name'])
        self.chmap = d['chmap']
        self.keystrings = d['keystrings']
        if 'format_version' in d:
            self._load_v2(d)
        else:
            self._load_legacy(d)

    # ---------- new format ----------
    def _load_v2(self, d):
        self.format_version = int(d['format_version'])
        self.legacy = False
        self.n_dishes = d['n_dishes']
        self.n_channels = d['n_channels']
        self.instrument = str(d['instrument']) or None
        self.chmap_to_dishes = d['chmap_to_dishes']
        self.dish_names = d['dish_names']
        self.origin = d['origin']
        self.origin_given = bool(d['origin_given'])
        cm = np.atleast_1d(d['coords_mode'])   # per dish ('enu' / 'geodetic')
        self.coords_mode = [str(x) for x in cm] if len(cm) > 1 or int(d['n_dishes']) == 1 \
            else [str(cm[0])] * int(d['n_dishes'])
        self.coords_input = d['coords_input']
        self.dish_coords = d['dish_coords']
        self.dish_coords_geo = d['dish_coords_geo']
        self.dish_pointing = d['dish_pointing']
        self.pol_labels = d['pol_labels']
        self.pol_angle = d['pol_angle']
        self.coords = d['coords']
        self.pointings = d['pointings']
        self.polarizations = d['polarizations']
        self.notes = str(d['notes'])

    # ---------- old format ----------
    def _load_legacy(self, d):
        """Old files: per-channel coords/pointings/polarizations, dish grouping
        never stored. Infer dishes as groups of channels with identical coords
        (each NaN-coord channel is its own instrument)."""
        self.format_version = 1
        self.legacy = True
        self.instrument = None
        self.notes = ''
        self.n_channels = d['n_channels']
        self.origin = np.asarray(d['origin'], float)
        self.origin_given = True
        self.coords_mode = 'enu'
        self.coords = np.asarray(d['coords'], float)
        self.pointings = np.asarray(d['pointings'], float)
        self.polarizations = np.asarray(d['polarizations'], float)
        self.n_dishes_stored = int(d['n_dishes'])

        n_ch = len(self.keystrings)
        dish_of, reps = [], []
        for i in range(n_ch):
            c = self.coords[i]
            match = None
            if not np.any(np.isnan(c)):
                for k, r in enumerate(reps):
                    if not np.any(np.isnan(self.coords[r])) and np.allclose(self.coords[r], c):
                        match = k
                        break
            if match is None:
                reps.append(i)
                match = len(reps) - 1
            dish_of.append(match)
        self.chmap_to_dishes = np.array(dish_of)
        self.n_dishes = np.array(len(reps))
        self.dish_names = np.array(_default_dish_names([str(k) for k in self.keystrings],
                                                       self.chmap_to_dishes, len(reps)))
        self.dish_coords = self.coords[reps]
        self.coords_input = self.dish_coords.copy()
        self.coords_mode = ['enu'] * len(reps)
        self.dish_coords_geo = np.array([enu_to_geo(c, self.origin) for c in self.dish_coords])
        pts = []
        for r in reps:
            v = self.pointings[r]
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n == 0:
                pts.append([np.nan, np.nan])
                continue
            v = v / n
            tilt = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
            az = np.degrees(np.arctan2(v[0], v[1])) % 360 if tilt > 1e-9 else 0.0
            pts.append([tilt, az])
        self.dish_pointing = np.array(pts)
        angs = []
        for i in range(n_ch):
            v, b = self.polarizations[i], self.pointings[i]
            nv, nb = np.linalg.norm(v), np.linalg.norm(b)
            # a real polarization is perpendicular to the boresight
            if nv == 0 or nb == 0 or abs(np.dot(v, b)) / (nv * nb) > 0.5:
                angs.append(np.nan)
            else:
                angs.append(np.degrees(np.arctan2(v[0], v[1])) % 360)
        self.pol_angle = np.array(angs)
        self.pol_labels = np.array(['None' if np.isnan(a) else '{:g}'.format(a) for a in angs])

    ## ---------- convenience ----------
    def channels_of_dish(self, dish):
        """Channel indices (positions in chmap/keystrings) for a dish index or name."""
        if isinstance(dish, str):
            dish = list(self.dish_names).index(dish)
        return np.where(self.chmap_to_dishes == dish)[0]

    def inputs(self):
        """Raw arguments that would recreate this site with make_site_file()."""
        return dict(
            name=self.name,
            n_dishes=int(self.n_dishes),
            n_channels=int(self.n_channels) if not self.legacy else len(self.keystrings),
            chmap=[int(x) for x in self.chmap],
            chmap_to_dishes=[int(x) for x in self.chmap_to_dishes],
            keystrings=[str(x) for x in self.keystrings],
            origin=[float(x) for x in self.origin] if self.origin_given else None,
            coords=self.coords_input.tolist(),
            coords_mode=['m' if m == 'enu' else 'latlon' for m in self.coords_mode],
            pointings=self.dish_pointing.tolist(),
            polarizations=[None if np.isnan(a) else
                           (str(l) if str(l).upper() in _COMPASS_DEG else float(a))
                           for a, l in zip(self.pol_angle, self.pol_labels)],
            instrument=self.instrument,
            dish_names=[str(x) for x in self.dish_names],
            notes=self.notes,
        )

    def show(self):
        print(_format_site(self))

    def __repr__(self):
        return _format_site(self)


############################################################
# CREATE / VALIDATE
############################################################
def _default_dish_names(keystrings, chmap_to_dishes, n_dishes):
    names = []
    for k in range(n_dishes):
        ks = [keystrings[i] for i in range(len(keystrings)) if chmap_to_dishes[i] == k]
        if len(ks) == 1:
            names.append(ks[0])
            continue
        # shared leading words, split on _ - space (CHORD_NW, CHORD_NE -> CHORD)
        toks = [re.split(r'([_\- ])', s) for s in ks]
        common = []
        for parts in zip(*toks):
            if all(p == parts[0] for p in parts):
                common.append(parts[0])
            else:
                break
        pre = ''.join(common).rstrip('_- ')
        names.append(pre if pre else 'dish{}'.format(k))
    return names


def _build(name, n_dishes, n_channels, chmap, chmap_to_dishes, keystrings,
           coords, origin=None, pointings='zenith', polarizations=None,
           instrument=None, dish_names=None, notes='', coords_mode='auto'):
    """Validate the inputs and return the dict of arrays to save."""
    err = []
    n_dishes, n_channels = int(n_dishes), int(n_channels)
    chmap = [int(x) for x in chmap]
    chmap_to_dishes = [int(x) for x in chmap_to_dishes]
    keystrings = [str(x) for x in keystrings]

    if len(chmap) != n_channels:
        err.append("chmap has {} entries but n_channels = {}".format(len(chmap), n_channels))
    if len(set(chmap)) != len(chmap):
        err.append("chmap lists a port more than once: {}".format(chmap))
    if instrument is not None:
        instrument = str(instrument).strip().lower()
        if instrument not in N_PORTS:
            err.append("instrument must be 'crs', 'ice' or None, got {!r}".format(instrument))
        else:
            bad = [p for p in chmap if not 0 <= p < N_PORTS[instrument]]
            if bad:
                err.append("ports {} don't exist on the {} (ports 0-{})".format(
                    bad, instrument.upper(), N_PORTS[instrument] - 1))
    elif any(p < 0 or p > 15 for p in chmap):
        err.append("chmap ports must be 0-7 (CRS) or 0-15 (ICEBoard): {}".format(chmap))
    if len(keystrings) != n_channels:
        err.append("keystrings has {} entries but n_channels = {}".format(len(keystrings), n_channels))
    elif len(set(keystrings)) != len(keystrings):
        err.append("keystrings must be unique: {}".format(keystrings))
    if len(chmap_to_dishes) != n_channels:
        err.append("chmap_to_dishes has {} entries but n_channels = {}".format(
            len(chmap_to_dishes), n_channels))
    else:
        bad = [d for d in chmap_to_dishes if not 0 <= d < n_dishes]
        if bad:
            err.append("chmap_to_dishes values must be 0-{} (n_dishes = {}): got {}".format(
                n_dishes - 1, n_dishes, bad))
        unused = [k for k in range(n_dishes) if k not in chmap_to_dishes]
        if unused:
            err.append("dish index(es) {} have no channels in chmap_to_dishes".format(unused))

    # ---- coords ----
    rows = []
    try:
        for r in list(coords):
            if r is None:
                rows.append([np.nan] * 3)
                continue
            r = np.asarray(r, float).ravel()
            if r.shape == (1,) and np.isnan(r[0]):
                r = np.full(3, np.nan)
            if r.shape != (3,):
                raise ValueError("each coords row needs 3 numbers, got {}".format(r.tolist()))
            rows.append(r)
        coords_in = np.array(rows, float).reshape(-1, 3)
        if len(coords_in) != n_dishes:
            err.append("coords has {} rows but n_dishes = {} (one row per dish, NaN row "
                       "if location doesn't matter)".format(len(coords_in), n_dishes))
    except (TypeError, ValueError) as e:
        err.append("coords: {}".format(e))
        coords_in = np.full((n_dishes, 3), np.nan)

    origin_given = origin is not None
    # per-dish coords_mode: 'enu' (metres from origin) or 'geodetic' (lat, lon, elev)
    if isinstance(coords_mode, str) or coords_mode is None:
        modes = [coords_mode or 'auto'] * len(coords_in)
    else:
        modes = list(coords_mode)
        if len(modes) != len(coords_in):
            err.append("coords_mode has {} entries but there are {} coords rows".format(
                len(modes), len(coords_in)))
            modes = (modes + ['auto'] * len(coords_in))[:len(coords_in)]
    norm = []
    for k, m in enumerate(modes):
        key = 'auto' if m is None else str(m).strip().lower()
        if key == 'auto':
            norm.append('enu' if origin_given else 'geodetic')
        elif key in _MODE_ALIASES:
            norm.append(_MODE_ALIASES[key])
        else:
            err.append("coords_mode {!r} for dish {} not understood; use 'm' (metres from "
                       "origin) or 'latlon' (lat, lon, elev)".format(m, k))
            norm.append('enu')
    coords_mode = norm
    if 'enu' in coords_mode and not origin_given:
        err.append("dish(es) {} are given in metres, so an origin [lat, lon, elev] is "
                   "needed".format([k for k, m in enumerate(coords_mode) if m == 'enu']))
    if origin_given:
        origin = np.asarray(origin, float).ravel()
        if origin.shape != (3,):
            err.append("origin should be [lat, lon, elev]")
            origin = np.full(3, np.nan)
        elif not (-90 <= origin[0] <= 90 and -180 <= origin[1] <= 360):
            err.append("origin {} doesn't look like [lat, lon, elev]".format(origin.tolist()))
    for k, r in enumerate(coords_in):
        # a "metres" row sitting within a degree of the origin's own lat AND lon is a lat/lon
        if coords_mode[k] == 'enu' and origin_given and np.asarray(origin).shape == (3,) \
                and not np.any(np.isnan(r)) and abs(r[0] - origin[0]) < 1 \
                and abs(r[1] - origin[1]) < 1:
            err.append("dish {} coords {} look like [lat, lon, elev], but this dish is being "
                       "read as metres from the origin - set coords_mode for it to "
                       "'latlon'".format(k, r.tolist()))
        if coords_mode[k] == 'geodetic' and not np.any(np.isnan(r)) \
                and not (-90 <= r[0] <= 90 and -180 <= r[1] <= 360):
            err.append("dish {} coords {} don't look like [lat, lon, elev] - if they are "
                       "metres from the origin, set coords_mode for it to 'm'".format(
                           k, r.tolist()))

    # ---- pointings ----
    try:
        if pointings is None or isinstance(pointings, str):
            dish_pointing = [_pointing_entry(pointings)] * n_dishes
        else:
            flat = np.asarray(pointings, dtype=object).ravel()
            if len(flat) == 2 and all(isinstance(x, (int, float, np.integer, np.floating))
                                      for x in flat):
                dish_pointing = [_pointing_entry(list(flat))] * n_dishes   # one for all
            else:
                pl = list(pointings)
                if len(pl) != n_dishes:
                    raise ValueError("{} entries but n_dishes = {}".format(len(pl), n_dishes))
                dish_pointing = [_pointing_entry(p) for p in pl]
        dish_pointing = np.array(dish_pointing, float)
    except ValueError as e:
        err.append("pointings: {}".format(e))
        dish_pointing = np.zeros((n_dishes, 2))

    # ---- polarizations ----
    if polarizations is None:
        polarizations = [None] * n_channels
    try:
        pol = [_pol_to_deg(p) for p in polarizations]
        if len(pol) != n_channels:
            err.append("polarizations has {} entries but n_channels = {}".format(len(pol), n_channels))
    except ValueError as e:
        err.append(str(e))
        pol = [(np.nan, 'None')] * n_channels

    if dish_names is None and not err:
        dish_names = _default_dish_names(keystrings, chmap_to_dishes, n_dishes)
    elif dish_names is not None:
        dish_names = [str(x) for x in dish_names]
        if len(dish_names) != n_dishes:
            err.append("dish_names has {} entries but n_dishes = {}".format(len(dish_names), n_dishes))

    if err:
        raise ValueError("Site file not valid:\n  - " + "\n  - ".join(err))

    # ---- derived coordinates (every dish gets both forms) ----
    if not origin_given:
        ok = [r for r in coords_in if not np.any(np.isnan(r))]
        origin = np.array(ok[0]) if ok else np.full(3, np.nan)
    dish_coords = np.zeros((len(coords_in), 3))
    dish_geo = np.zeros((len(coords_in), 3))
    for k, (r, m) in enumerate(zip(coords_in, coords_mode)):
        if m == 'enu':
            dish_coords[k], dish_geo[k] = r, enu_to_geo(r, origin)
        else:
            dish_geo[k], dish_coords[k] = r, geo_to_enu(r, origin)

    pol_angle = np.array([p[0] for p in pol])
    ch_coords = np.array([dish_coords[d] for d in chmap_to_dishes])
    ch_point = np.array([pointing_vector(*dish_pointing[d]) for d in chmap_to_dishes])
    ch_pol = np.array([pol_vector(pol_angle[i], *dish_pointing[d])
                       for i, d in enumerate(chmap_to_dishes)])
    return dict(
        format_version=np.array(FORMAT_VERSION),
        name=np.array(str(name)),
        instrument=np.array(instrument or ''),
        n_dishes=np.array(n_dishes),
        n_channels=np.array(n_channels),
        chmap=np.array(chmap, dtype=int),
        chmap_to_dishes=np.array(chmap_to_dishes, dtype=int),
        keystrings=np.array(keystrings),
        dish_names=np.array(dish_names),
        origin=np.asarray(origin, float),
        origin_given=np.array(origin_given),
        coords_mode=np.array(coords_mode).reshape(-1),
        coords_input=coords_in,
        dish_coords=dish_coords,
        dish_coords_geo=dish_geo,
        dish_pointing=dish_pointing,
        pol_labels=np.array([p[1] for p in pol]),
        pol_angle=pol_angle,
        coords=ch_coords,          # legacy per-channel arrays, same meaning as old files
        pointings=ch_point,
        polarizations=ch_pol,
        notes=np.array(str(notes)),
    )


def _resolve_path(path, default_name):
    """path may be None (current folder), a folder, or a full file path.
    A path ending in / or naming an existing folder is treated as a folder."""
    if path is None or path == '':
        path = '.'
    path = os.path.expanduser(str(path))
    if path.endswith(('/', os.sep)) or os.path.isdir(path):
        path = os.path.join(path, default_name)
    if not path.endswith('.npz'):
        path += '.npz'
    return path


def _save(arrays, filename, overwrite):
    if not filename.endswith('.npz'):
        filename += '.npz'
    folder = os.path.dirname(filename)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
        print("Created folder {}".format(folder))
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError("{} already exists. Use overwrite=True, or "
                              "edit_site_file() to change it.".format(filename))
    np.savez(filename, **arrays)
    return filename


def make_site_file(name, n_dishes, n_channels, chmap, chmap_to_dishes, keystrings,
                   coords, origin=None, pointings='zenith', polarizations=None,
                   instrument=None, dish_names=None, notes='', save_path=None,
                   overwrite=False, coords_mode='auto', show=True,
                   filename=None, directory=None):
    """Create, validate and save a site file. Returns the loaded Site.

    save_path       where to save: None = current folder as <name>.npz; a folder
                    ('/data/site_files/') = <name>.npz in that folder; or a full
                    file path ('/data/site_files/flight3.npz'). Missing folders
                    are created. (filename= / directory= still work too.)
    name            what you call this dataset (default file name is <name>.npz)
    n_dishes        number of independent instruments (dishes, reference boards, antennas...)
    n_channels      number of correlator inputs you care about
    chmap           which port carried each channel (CRS 0-7, ICEBoard 0-15)
    chmap_to_dishes which dish (0..n_dishes-1) each channel belongs to
    keystrings      your name for each channel, same order as chmap
    coords          one row per dish. With origin: [East, North, Up] metres from it.
                    Without origin: [lat, lon, elev]. NaN row (or None) = irrelevant.
    origin          [lat, lon, elev] reference point, or None
    coords_mode     'auto' (default: metres if origin given, else lat/lon), or one
                    entry per dish, 'm' or 'latlon', to mix the two, e.g.
                    coords_mode=['m', 'm', 'm', 'latlon', 'latlon', 'latlon']
    pointings       'zenith' (default, all dishes), one [tilt, azimuth] for all dishes,
                    or one entry per dish ('zenith' or [tilt, azimuth]); degrees,
                    tilt from zenith, azimuth from North towards East
    polarizations   one per channel: degrees from North towards East, compass name
                    ('N', 'NE', 'E', 'NW', 'NS', 'EW', ...), or None
    instrument      optional 'crs'/'ice' - only used to check the port numbers
    dish_names      optional; default from shared keystring prefix (CHORD_NW/CHORD_NE -> CHORD)
    """
    arrays = _build(name, n_dishes, n_channels, chmap, chmap_to_dishes, keystrings, coords,
                    origin=origin, pointings=pointings, polarizations=polarizations,
                    instrument=instrument, dish_names=dish_names, notes=notes,
                    coords_mode=coords_mode)
    if save_path is None and (filename or directory):
        save_path = os.path.join(directory or '.', filename or '')
    path = _save(arrays, _resolve_path(save_path, str(name) + '.npz'), overwrite)
    site = Site(path)
    print("Saved site file: {}".format(path))
    if show:
        site.show()
    return site


def edit_site_file(filename, save_as=None, backup=True, show=True, **changes):
    """Change part of an existing site file (old or new format) and save it.

    Any make_site_file() argument can be changed. For per-item fields
    (chmap, chmap_to_dishes, keystrings, polarizations, coords, pointings,
    dish_names) you can pass a dict {index: new_value} to change single
    entries, e.g.:
        edit_site_file('GBO_DNS_config.npz', polarizations={2: 'NW', 3: 'NE'})
        edit_site_file('GBO_DNS_config.npz', coords={1: [-16.5, 21.9, 0]})
        edit_site_file('old.npz', save_as='new.npz', n_dishes=3,
                       chmap_to_dishes=[0, 1, 2, 2])
    An index one past the end ADDS an entry (new channels / dishes).
    n_channels and n_dishes follow the length of chmap and coords unless
    you give them. New dishes default to zenith pointing and new channels to
    no polarization. A new dish given as lat/lon in a file with an origin
    needs coords_mode={its index: 'latlon'}.
    By default the original is overwritten after copying it to
    <file>_backup.npz; give save_as to write a new file instead. save_as can
    be a full file path, or a folder (the original file name is kept), e.g.
    save_as='/data/site_files/'. Missing folders are created.
    Old-format files are always saved in the new format.
    """
    site = Site(filename)
    args = site.inputs()
    if site.legacy:
        print("NOTE: {} is an old-format site file. Inferred from identical coordinates: "
              "{} dish(es), chmap_to_dishes = {}{}. Polarizations stored in old files are "
              "not usable and were set to None - set them if they matter.".format(
                  filename, args['n_dishes'], args['chmap_to_dishes'],
                  " (file said n_dishes = {})".format(site.n_dishes_stored)
                  if site.n_dishes_stored != args['n_dishes'] else ''))
    regen_names = 'dish_names' not in changes and any(
        k in changes for k in ('chmap_to_dishes', 'n_dishes', 'keystrings', 'coords'))
    # a whole new coords list with no coords_mode -> work the format out from origin again
    if 'coords' in changes and not isinstance(changes['coords'], dict) \
            and 'coords_mode' not in changes:
        args['coords_mode'] = 'auto'
    if isinstance(args.get('pointings'), str):
        args['pointings'] = [args['pointings']] * int(args['n_dishes'])

    _MISSING = object()
    for key, val in changes.items():
        if key not in args:
            raise TypeError("Unknown site field {!r}. Fields: {}".format(key, list(args)))
        if isinstance(val, dict):
            if not isinstance(args[key], list):
                raise TypeError("{} isn't a per-item field; give the whole value".format(key))
            new = list(args[key])
            for i, v in sorted(val.items()):
                i = int(i)
                while len(new) <= i:          # index past the end -> add entries
                    new.append(_MISSING)
                new[i] = v
            args[key] = new
        else:
            args[key] = val

    # counts follow the lists unless given explicitly
    if 'n_channels' not in changes and len(args['chmap']) != int(args['n_channels']):
        print("n_channels: {} -> {}".format(args['n_channels'], len(args['chmap'])))
        args['n_channels'] = len(args['chmap'])
    if 'n_dishes' not in changes and len(args['coords']) != int(args['n_dishes']):
        print("n_dishes: {} -> {}".format(args['n_dishes'], len(args['coords'])))
        args['n_dishes'] = len(args['coords'])
    n_ch, n_d = int(args['n_channels']), int(args['n_dishes'])

    # fill sensible defaults for anything not given for new channels / dishes
    def _pad(key, n, fill):
        cur = args[key] if isinstance(args[key], list) else [args[key]] * n
        cur = [fill if x is _MISSING else x for x in cur]
        args[key] = cur + [fill] * (n - len(cur))
    _pad('polarizations', n_ch, None)
    _pad('pointings', n_d, 'zenith')
    if not isinstance(args['coords_mode'], str):
        _pad('coords_mode', n_d, 'auto')
    missing = [k for k in ('chmap', 'chmap_to_dishes', 'keystrings', 'coords')
               if any(x is _MISSING for x in args[k])]
    if missing:
        raise ValueError("Gaps left in {} - when adding entries by index, give every new "
                         "index (no skipping).".format(missing))
    if regen_names:
        args['dish_names'] = None
    if 'origin' in changes and changes['origin'] is None:
        modes = [args['coords_mode']] * n_d if isinstance(args['coords_mode'], str) \
            else args['coords_mode']
        if any(str(m).lower() in ('m', 'enu') for m in modes):
            raise ValueError("Removing the origin would leave dishes given in metres with "
                             "nothing to measure from. Give those dishes as [lat, lon, elev] "
                             "with coords_mode='latlon' in the same call.")
    out = _resolve_path(save_as, os.path.basename(filename)) if save_as else filename
    arrays = _build(**args)
    if os.path.abspath(out) == os.path.abspath(filename) and backup:
        root = filename[:-4] if filename.endswith('.npz') else filename
        shutil.copy2(filename if filename.endswith('.npz') else filename + '.npz',
                     root + '_backup.npz')
        print("Backed up original to {}_backup.npz".format(root))
    path = _save(arrays, out, overwrite=True)
    print("Saved site file: {}".format(path))
    new_site = Site(path)
    if show:
        new_site.show()
    return new_site


def show_site_file(filename):
    """Pretty-print any site file (old or new format)."""
    Site(filename).show()


############################################################
# DISPLAY
############################################################
def _fmt(x, nd=2):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return '-'
    s = '{:.{}f}'.format(x, nd)
    return s[1:] if re.fullmatch(r'-0\.0*', s) else s


def _table(hdr, rows):
    w = [max(len(h), *(len(r[c]) for r in rows)) if rows else len(h) for c, h in enumerate(hdr)]
    line = '  '.join('{:<%d}' % x for x in w)
    out = [line.format(*hdr), '  '.join('-' * x for x in w)]
    out += [line.format(*r) for r in rows]
    return '\n'.join('  ' + s for s in out)


def _format_site(s):
    L = []
    L.append("Site: {}".format(s.name))
    L.append("  file: {}   format: {}{}".format(
        s.filename, 'v{}'.format(s.format_version),
        ' (old format; dishes inferred from shared coords)' if s.legacy else ''))
    if s.instrument:
        L.append("  instrument: {}".format(s.instrument.upper()))
    o = s.origin
    if np.any(np.isnan(o)):
        L.append("  origin: none")
    else:
        L.append("  origin: lat {:.6f}, lon {:.6f}, elev {:.1f} m{}".format(
            o[0], o[1], o[2], '' if s.origin_given else '  (taken from first located dish)'))
    modes = set(s.coords_mode)
    L.append("  coords given as: {}".format(
        'East/North/Up metres from origin' if modes == {'enu'} else
        'lat/lon/elev per dish' if modes == {'geodetic'} else
        'mixed (see "given as" column)'))
    if s.legacy and int(s.n_dishes) != s.n_dishes_stored:
        L.append("  NOTE: file says n_dishes = {}, channels share coords like {} dishes".format(
            s.n_dishes_stored, int(s.n_dishes)))

    L.append("")
    L.append("Dishes / instruments ({}):".format(int(s.n_dishes)))
    rows = []
    for k in range(int(s.n_dishes)):
        ch = [str(s.keystrings[i]) for i in np.where(s.chmap_to_dishes == k)[0]]
        e, n, u = s.dish_coords[k]
        la, lo, el = s.dish_coords_geo[k]
        t, a = s.dish_pointing[k]
        if np.isnan(t):
            pt = '-'
        elif t == 0:
            pt = 'zenith'
        else:
            pt = '{:g} deg toward {:g}'.format(t, a)
        given = '-' if np.all(np.isnan(s.dish_coords_geo[k])) else \
            ('m' if s.coords_mode[k] == 'enu' else 'lat/lon')
        rows.append((str(k), str(s.dish_names[k]), ', '.join(ch),
                     _fmt(e), _fmt(n), _fmt(u), _fmt(la, 6), _fmt(lo, 6), _fmt(el, 1),
                     given, pt))
    L.append(_table(('idx', 'name', 'channels', 'E (m)', 'N (m)', 'Up (m)',
                     'lat', 'lon', 'elev', 'given as', 'pointing'), rows))

    L.append("")
    L.append("Channels ({}):".format(len(s.keystrings)))
    rows = []
    for i in range(len(s.keystrings)):
        a, lab = s.pol_angle[i], str(s.pol_labels[i])
        if np.isnan(a):
            pol = '-'
        elif lab.upper() in _COMPASS_DEG:
            pol = '{} ({:g} deg)'.format(lab, a)
        else:
            pol = '{:g} deg'.format(a)
        d = int(s.chmap_to_dishes[i])
        rows.append((str(i), str(s.keystrings[i]), str(int(s.chmap[i])),
                     '{} ({})'.format(d, s.dish_names[d]), pol))
    L.append(_table(('idx', 'keystring', 'port', 'dish', 'polarization'), rows))
    if s.notes:
        L.append("")
        L.append("Notes: " + s.notes)
    return '\n'.join(L)


############################################################
# TEMPLATE
############################################################
_TEMPLATE = '''import numpy as np
import site_utils

# whatever you want to call this dataset + _config (also the file name)
name = 'MY_DATASET_config'

# 'crs' (ports 0-7) or 'ice' (ports 0-15); only used to check chmap
instrument = 'crs'

# total number of independent instruments measured (dishes, reference board, antennas...)
n_dishes = 3

# how many correlator inputs you actually care about
n_channels = 4

# which port carried each of those channels
chmap = [0, 1, 2, 3]

# which dish (0 .. n_dishes-1) each channel in chmap belongs to
chmap_to_dishes = [0, 1, 2, 2]

# what you want to call each channel, same order as chmap
keystrings = ['Reference', 'BicoLOG', 'CHORD_NW', 'CHORD_NE']

# reference point [lat, lon, elev in m]. If given, coords below are
# [East, North, Up] metres from it. Set origin = None to give every dish
# as [lat, lon, elev] instead.
origin = [38.42928, -79.84499, 810.]

# one row per dish, in dish order. np.nan row = location irrelevant
coords = [[np.nan, np.nan, np.nan],
          [-16.47562334, 21.89848478, 0.],
          [-16.47562334, 11.94168478, 0.]]

# how each coords row is given: 'auto' (metres if origin is set, otherwise
# lat/lon), or one per dish, 'm' or 'latlon', to mix them
coords_mode = 'auto'

# 'zenith' for all dishes, or one per dish: 'zenith' or [tilt from zenith, azimuth]
# in degrees, azimuth measured from North towards East
pointings = 'zenith'

# one per channel: degrees from North towards East, or a compass direction
# ('N','NE','E','NW','NS','EW',...), or None. (NE == SW: it's an axis.)
polarizations = [None, None, 'NW', 'NE']

# where to save: None = current folder as <name>.npz, a folder ('/path/to/site_files/'),
# or a full file path ('/path/to/site_files/my_site.npz'). Missing folders are created.
save_path = None

site = site_utils.make_site_file(save_path=save_path,name=name, instrument=instrument, n_dishes=n_dishes,
                                 n_channels=n_channels, chmap=chmap,
                                 chmap_to_dishes=chmap_to_dishes, keystrings=keystrings,
                                 origin=origin, coords=coords, coords_mode=coords_mode,
                                 pointings=pointings,
                                 polarizations=polarizations)
'''


def site_template():
    """Print a fill-in-the-blanks notebook cell for make_site_file()."""
    print(_TEMPLATE)


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        show_site_file(f)
        print()