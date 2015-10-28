'''
Procedures for reading/writing header metadata of exposures.
'''
from __future__ import absolute_import
import re
import dateutil.parser
import datetime
import numpy as np
import gzip
import math
import functools
import os
import sys
if sys.version_info[0] == 2:
    from io import open


from .. import libconfig
from .. import misc


def _linearize_bool(b):
    return ['n', 'y'][bool(b)]


def _delinearize_bool(b):
    if isinstance(b, str):
        b = b.lower()
        if b.startswith('y') or b == 'true':
            return True
        elif b.startswith('n') or b == 'false':
            return False
        else:
            raise ValueError(b)
    else:
        return bool(b)


def _linearize_list(l, pre_converter=lambda a: a, post_converter=lambda a: a):
    return post_converter(' '.join([str(pre_converter(x)) for x in l]))


def _delinearize_list(l, pre_converter=lambda a: a, post_converter=list):
    return post_converter([misc.parse_number(x) for x in
                           pre_converter(l).replace(',', ' ').replace(';', ' ').split()])


def _linearize_history(history):
    history_text = [str(x[0]) + ': ' + x[1] for x in history]
    history_text = [a.replace(';', ';;') for a in history_text]
    return '; '.join(history_text)


def _delinearize_history(history_oneliner):
    history_oneliner = history_oneliner.replace(';;',
                                                '<*doublesemicolon*>')
    history_list = [a.strip().replace('<*doublesemicolon*>', ';')
                    for a in history_oneliner.split(';')]
    history = [a.split(':', 1) for a in history_list]
    history = [(float(a[0]), a[1].strip()) for a in history]
    return history


# information on how to store the param structure. Each sub-list
# corresponds to a line in the param structure and should be of the form
# [<linestart>,<field name(s)>,<formatter function>,<reader function>]
#
# Required are the first and second.
#
# linestart: the beginning of the line in the file, up to the colon.
#
# field name(s): field name can be a string or a tuple of strings.
#
# formatter function: can be (1) a function accepting a single argument
#     (the value of the field) or (2) a tuple of functions or (3) None. In
#     the latter case and when omitted, lambda x:unicode(x).encode('utf-8')
#      will be used.
#
# reader function: can be (1) a function accepting a string and returning
#     as many values as the number of field names is. Or if omitted,
#     lambda x: str(x, encoding='utf-8') will be used.
#
_logfile_data = [('FSN', 'FSN', None, lambda x: int(float(x))),
                 ('FSNs', 'FSNs', _linearize_list, _delinearize_list),
                 ('Sample name', 'Title'),
                 ('Sample title', 'Title'),
                 ('Sample-to-detector distance (mm)', 'Dist', None, float),
                 ('Calibrated sample-to-detector distance (mm)',
                  'DistCalibrated', None, float),
                 ('Sample thickness (cm)', 'Thickness', None, float),
                 ('Sample transmission', 'Transm', None, float),
                 ('Sample position (mm)', 'PosSample', None, float),
                 ('Temperature', 'Temperature', None, float),
                 ('Measurement time (sec)', 'MeasTime', None, float),
                 ('Scattering on 2D detector (photons/sec)',
                  'ScatteringFlux', None, float),
                 ('Dark current subtracted (cps)', 'dclevel', None, float),
                 ('Dark current FSN', 'FSNdc', None, lambda x: int(float(x))),
                 ('Empty beam FSN', 'FSNempty',
                  None, lambda x: int(float(x))),
                 ('Injection between Empty beam and sample measurements?',
                  'InjectionEB', _linearize_bool, _delinearize_bool),
                 ('Glassy carbon FSN', 'FSNref1',
                  None, lambda x: int(float(x))),
                 ('Glassy carbon thickness (cm)', 'Thicknessref1', None,
                  float),
                 ('Injection between Glassy carbon and sample measurements?',
                  'InjectionGC', _linearize_bool, _delinearize_bool),
                 ('Energy (eV)', 'Energy', None, float),
                 ('Calibrated energy (eV)', 'EnergyCalibrated', None, float),
                 ('Calibrated energy', 'EnergyCalibrated', None, float),
                 ('Beam x y for integration', ('BeamPosX', 'BeamPosY'),
                  functools.partial(
                      _linearize_list, pre_converter=lambda a:a + 1),
                  functools.partial(_delinearize_list,
                                    post_converter=lambda a:tuple([x - 1 for x in a]))),
                 ('Normalisation factor (to absolute units)', 'NormFactor',
                  None, float),
                 ('Relative error of normalisation factor (percentage)',
                  'NormFactorRelativeError', None, float),
                 ('Beam size X Y (mm)', ('BeamsizeX', 'BeamsizeY'), _linearize_list,
                  functools.partial(_delinearize_list, post_converter=tuple)),
                 ('Pixel size of 2D detector (mm)', 'PixelSize', None, float),
                 ('Primary intensity at monitor (counts/sec)', 'Monitor', None,
                  float),
                 ('Primary intensity calculated from GC (photons/sec/mm^2)',
                  'PrimaryIntensity', None, float),
                 ('Sample rotation around x axis', 'RotXsample', None, float),
                 ('Sample rotation around y axis', 'RotYsample', None, float),
                 ('History', 'History', _linearize_history,
                  _delinearize_history),
                 ('Date', 'Date', None, dateutil.parser.parse),
                 ('EndDate', 'EndDate', None, dateutil.parser.parse),
                 ('CBF_Date', 'CBF_Date', None, dateutil.parser.parse),
                 ]


def readB1logfile(filename):
    """Read B1 logfile (*.log)

    Inputs:
        filename: the file name

    Output: A dictionary.
    """
    dic = dict()
    # try to open. If this fails, an exception is raised
    with open(filename, 'rt', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l[0] in '#!%\'':
                continue  # treat this line as a comment
            try:
                # find the first tuple in _logfile_data where the first element of the
                # tuple is the starting of the line.
                ld = [ld_ for ld_ in _logfile_data if l.split(
                    ':', 1)[0].strip() == ld_[0]][0]
            except IndexError:
                # line is not recognized. We can still try to load it: find the first
                # semicolon. If found, the part of the line before it is stripped
                # from whitespaces and will be the key. The part after it is stripped
                # from whitespaces and parsed with misc.parse_number(). If no
                if ':' in l:
                    key, val = [x.strip() for x in l.split(':', 1)]
                    val = misc.parse_number(val)
                    dic[key] = val
                    try:
                        # fix the character encoding in files written by a
                        # previous version of this software.
                        dic[key] = dic[key].encode('latin2').decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
                        pass
                else:
                    dic[l.strip()] = True
                continue
            try:
                reader = ld[3]
            except IndexError:
                reader = str
            rhs = l.split(':', 1)[1].strip()
            try:
                vals = reader(rhs)
            except ValueError:
                if rhs.lower() == 'none':
                    vals = None
                else:
                    raise
            if isinstance(ld[1], tuple):
                # more than one field names. The reader function should return a
                # tuple here, a value for each field.
                if len(vals) != len(ld[1]):
                    raise ValueError(
                        'Cannot read %d values from line %s in file!' % (len(ld[1]), l))
                dic.update(dict(list(zip(ld[1], vals))))
            else:
                dic[ld[1]] = vals
    dic['__Origin__'] = 'B1 log'
    dic['__particle__'] = 'photon'
    return dic


def writeB1logfile(filename, data):
    """Write a header structure into a B1 logfile.

    Inputs:
        filename: name of the file.
        data: header dictionary

    Notes:
        exceptions pass through to the caller.
    """
    allkeys = list(data.keys())
    f = open(filename, 'wt', encoding='utf-8')
    for ld in _logfile_data:  # process each line
        linebegin = ld[0]
        fieldnames = ld[1]
        # set the default formatter if it is not given
        if len(ld) < 3:
            formatter = str
        elif ld[2] is None:
            formatter = str
        else:
            formatter = ld[2]
        # this will contain the formatted values.
        formatted = ''
        if isinstance(fieldnames, str):
            # scalar field name, just one field. Formatter should be a
            # callable.
            if fieldnames not in allkeys:
                # this field has already been processed
                continue
            try:
                formatted = formatter(data[fieldnames])
            except KeyError:
                # field not found in param structure
                continue
        elif isinstance(fieldnames, tuple):
            # more than one field names in a tuple. In this case, formatter can
            # be a tuple of callables...
            if all([(fn not in allkeys) for fn in fieldnames]):
                # if all the fields have been processed:
                continue
            if isinstance(formatter, tuple) and len(formatter) == len(fieldnames):
                formatted = ' '.join([ft(data[fn])
                                      for ft, fn in zip(formatter, fieldnames)])
            # ...or a single callable...
            elif not isinstance(formatter, tuple):
                formatted = formatter([data[fn] for fn in fieldnames])
            # ...otherwise raise an exception.
            else:
                raise SyntaxError('Programming error: formatter should be a scalar or a tuple\
of the same length as the field names in logfile_data.')
        else:  # fieldnames is neither a string, nor a tuple.
            raise SyntaxError(
                'Invalid syntax (programming error) in logfile_data in writeparamfile().')
        # try to get the values
        linetowrite = linebegin + ':\t' + formatted + '\n'
        if sys.version_info[0] == 2:
            if not isinstance(linetowrite, unicode):
                linetowrite = linetowrite.decode('utf-8')
        f.write(linetowrite)
        if isinstance(fieldnames, tuple):
            for fn in fieldnames:  # remove the params treated.
                if fn in allkeys:
                    allkeys.remove(fn)
        else:
            if fieldnames in allkeys:
                allkeys.remove(fieldnames)
    # write untreated params
    for k in allkeys:
        linetowrite = k + ':\t' + str(data[k]) + '\n'
        if sys.version_info[0] == 2:
            if not isinstance(linetowrite, unicode):
                linetowrite = linetowrite.decode('utf-8')
        f.write(linetowrite)

    f.close()


def readB1header(filename):
    """Read beamline B1 (HASYLAB, Hamburg) header data

    Input
    -----
    filename: string
        the file name. If ends with ``.gz``, it is fed through a ``gunzip``
        filter

    Output
    ------
    A header dictionary.

    Examples
    --------
    read header data from 'ORG000123.DAT'::

        header=readB1header('ORG00123.DAT')
    """
    # Planck's constant times speed of light: incorrect
    # constant in the old program on hasjusi1, which was
    # taken over by the measurement program, to keep
    # compatibility with that.
    hed = {}
    if libconfig.LENGTH_UNIT == 'A':
        jusifaHC = 12396.4
    elif libconfig.LENGTH_UNIT == 'nm':
        jusifaHC = 1239.64
    else:
        raise NotImplementedError(
            'Invalid length unit: ' + str(libconfig.LENGTH_UNIT))

    if filename.upper().endswith('.GZ'):
        fid = gzip.GzipFile(filename, 'r')
    else:
        fid = open(filename, 'rt')
    lines = fid.readlines()
    fid.close()
    hed['FSN'] = int(lines[0].strip())
    hed['Hour'] = int(lines[17].strip())
    hed['Minutes'] = int(lines[18].strip())
    hed['Month'] = int(lines[19].strip())
    hed['Day'] = int(lines[20].strip())
    hed['Year'] = int(lines[21].strip()) + 2000
    hed['FSNref1'] = int(lines[23].strip())
    hed['FSNdc'] = int(lines[24].strip())
    hed['FSNsensitivity'] = int(lines[25].strip())
    hed['FSNempty'] = int(lines[26].strip())
    hed['FSNref2'] = int(lines[27].strip())
    hed['Monitor'] = float(lines[31].strip())
    hed['Anode'] = float(lines[32].strip())
    hed['MeasTime'] = float(lines[33].strip())
    hed['Temperature'] = float(lines[34].strip())
    hed['BeamPosX'] = float(lines[36].strip())
    hed['BeamPosY'] = float(lines[37].strip())
    hed['Transm'] = float(lines[41].strip())
    hed['Wavelength'] = float(lines[43].strip())
    hed['Energy'] = jusifaHC / hed['Wavelength']
    hed['Dist'] = float(lines[46].strip())
    hed['XPixel'] = 1 / float(lines[49].strip())
    hed['YPixel'] = 1 / float(lines[50].strip())
    hed['Title'] = lines[53].strip().replace(' ', '_').replace('-', '_')
    hed['MonitorDORIS'] = float(lines[56].strip())  # aka. DORIS counter
    hed['Owner'] = lines[57].strip()
    hed['RotXSample'] = float(lines[59].strip())
    hed['RotYSample'] = float(lines[60].strip())
    hed['PosSample'] = float(lines[61].strip())
    hed['DetPosX'] = float(lines[62].strip())
    hed['DetPosY'] = float(lines[63].strip())
    hed['MonitorPIEZO'] = float(lines[64].strip())  # aka. PIEZO counter
    hed['BeamsizeX'] = float(lines[66].strip())
    hed['BeamsizeY'] = float(lines[67].strip())
    hed['PosRef'] = float(lines[70].strip())
    hed['Monochromator1Rot'] = float(lines[77].strip())
    hed['Monochromator2Rot'] = float(lines[78].strip())
    hed['Heidenhain1'] = float(lines[79].strip())
    hed['Heidenhain2'] = float(lines[80].strip())
    hed['Current1'] = float(lines[81].strip())
    hed['Current2'] = float(lines[82].strip())
    hed['Detector'] = 'Unknown'
    hed['PixelSize'] = (hed['XPixel'] + hed['YPixel']) / 2.0

    hed['AnodeError'] = math.sqrt(hed['Anode'])
    hed['TransmError'] = 0
    hed['MonitorError'] = math.sqrt(hed['Monitor'])
    hed['MonitorPIEZOError'] = math.sqrt(hed['MonitorPIEZO'])
    hed['MonitorDORISError'] = math.sqrt(hed['MonitorDORIS'])
    hed['Date'] = datetime.datetime(
        hed['Year'], hed['Month'], hed['Day'], hed['Hour'], hed['Minutes'])
    hed['__Origin__'] = 'B1 original'
    hed['__particle__'] = 'photon'
    return hed


def _readedf_extractline(left, right):
    """Helper function to interpret lines in an EDF file header.
    """
    functions = [int, float, lambda l:float(l.split(None, 1)[0]),
                 lambda l:int(l.split(None, 1)[0]),
                 dateutil.parser.parse, lambda x:str(x)]
    for f in functions:
        try:
            right = f(right)
            break
        except ValueError:
            continue
    return right


def readehf(filename):
    """Read EDF header (ESRF data format, as of beamline ID01 and ID02)

    Input
    -----
    filename: string
        the file name to load

    Output
    ------
    the EDF header structure in a dictionary
    """
    f = open(filename, 'r')
    edf = {}
    if not f.readline().strip().startswith('{'):
        raise ValueError('Invalid file format.')
    for l in f:
        l = l.strip()
        if not l:
            continue
        if l.endswith('}'):
            break  # last line of header
        try:
            left, right = l.split('=', 1)
        except ValueError:
            raise ValueError('Invalid line: ' + l)
        left = left.strip()
        right = right.strip()
        if not right.endswith(';'):
            raise ValueError(
                'Invalid line (does not end with a semicolon): ' + l)
        right = right[:-1].strip()
        m = re.match('^(?P<left>.*)~(?P<continuation>\d+)$', left)
        if m is not None:
            edf[m.group('left')] = edf[m.group('left')] + right
        else:
            edf[left] = _readedf_extractline(left, right)
    f.close()
    edf['FileName'] = filename
    edf['__Origin__'] = 'EDF ID02'
    edf['__particle__'] = 'photon'
    return edf


def readbhfv2(filename, load_data=False, bdfext='.bdf', bhfext='.bhf'):
    # strip the bhf or bdf extension if there.
    if filename.endswith(bdfext):
        basename = filename[:-len(bdfext)]
    elif filename.endswith(bhfext):
        basename = filename[:-len(bhfext)]
    else:  # assume a single file of header and data.
        basename, bhfext = os.path.splitext(filename)
        bdfext = bhfext
    headername = basename + bhfext
    dataname = basename + bdfext
    header = {}
    with open(headername, 'rt') as f:
        for l in f:
            if not l.startswith('#'):
                continue
            l = l[1:].strip()
            section, keyvalue = l.split(None, 1)
            if section not in list(header.keys()):
                if section in ['HIS']:
                    header[section] = []
                else:
                    header[section] = {}
            if section in ['HIS']:
                header[section].append(keyvalue)
            else:
                key, value = keyvalue.split('=')
                value = value.strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                header[section][key.strip()] = value
    header['xdim'] = header['C']['xdim']
    header['ydim'] = header['C']['ydim']
    header['type'] = header['C']['type']
    header['__Origin__'] = 'BDFv2'
    header['__particle__'] = 'photon'

    # now open the data if needed
    if load_data:
        f = open(dataname, 'rb')
        try:
            s = f.read()
        except IOError as ioe:
            # an ugly bug (M$ KB899149) in W!nd0w$ causes an error if loading too
            # large a file from a network drive and opening it read-only.
            if ioe.errno == 22:
                f.close()
                try:
                    # one work-around is to open it read-write.
                    f = open(dataname, 'r+b')
                    s = f.read()
                except IOError:
                    # if this does not work, inform the user to either obtain
                    # write permission for that file or copy it to a local
                    # drive
                    f.close()
                    raise IOError(22, """
You were probably trying to open a read-only file from a network drive on
Windows, weren\'t you? There is a bug in Windows causing this error
(see http://support.microsoft.com/default.aspx?scid=kb;en-us;899149).
To work around this, please either obtain write permission for that file
(I won't write anything to it, I promise!!!) or copy it to a local drive.
Sorry for the inconvenience.""", ioe.filename)
        datasets = re.findall(
            '#\s*(?P<name>\w+)\[(?P<xsize>\d+):(?P<ysize>\d+)\]', s)
        names = [d[0] for d in datasets]
        xsize = [int(d[1]) for d in datasets]
        ysize = [int(d[2]) for d in datasets]
        dt = np.dtype(header['type'])
        for i in range(len(datasets)):
            start = s.find('#%s' % names[i])
            if i < len(datasets) - 1:
                end = s.find('#%s' % (names[i + 1]))
            else:
                end = len(s)
            s1 = s[start:end]
            datasize = xsize[i] * ysize[i] * dt.itemsize
            if datasize > len(s1):
                # assume we are dealing with a BOOL matrix
                header[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i]:], dtype=np.uint8)
            else:
                header[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i] * dt.itemsize:], dtype=dt)
            # conversion: Matlab saves the array in Fortran-style ordering (columns first).
            # Python however loads in C-style: rows first. We need to take care:
            #   1) reshape from linear to (ysize,xsize) and not (xsize,ysize)
            #   2) transpose (swaps columns and rows)
            # After these operations, we only have to rotate this counter-clockwise by 90
            # degrees because bdf2_write rotates by +270 degrees before saving.
            header[names[i]] = np.rot90(
                header[names[i]].reshape((ysize[i], xsize[i]), order='F'), 1)
    return header


def writebhfv2(filename, bdf):
    f = open(filename, 'wt')
    f.write('#C xdim = %d\n' % bdf['xdim'])
    f.write('#C ydim = %d\n' % bdf['ydim'])
    f.write('#C type = %s\n' % bdf['type'])
    for k in [x for x in list(bdf.keys()) if isinstance(bdf[x], dict)]:
        f.write('-------------------- %s field --------------------\n' % k)
        for l in list(bdf[k].keys()):
            f.write("#%s %s = %s\n" % (k, l, bdf[k][l]))
    if 'HIS' in list(bdf.keys()):
        f.write('-------------------- History --------------------\n')
        for h in bdf['HIS']:
            f.write("#HIS %s\n" % h)
    f.close()


def readbhfv1(filename, load_data=False, bdfext='.bdf', bhfext='.bhf'):
    """Read header data from bdf/bhf file (Bessy Data Format v1)

    Input:
        filename: the name of the file
        load_data: if the matrices are to be loaded

    Output:
        bdf: the BDF header structure

    Adapted the bdf_read.m macro from Sylvio Haas.
    """
    # strip the bhf or bdf extension if there.
    if filename.endswith(bdfext):
        basename = filename[:-len(bdfext)]
    elif filename.endswith(bhfext):
        basename = filename[:-len(bhfext)]
    else:  # assume a single file of header and data.
        basename, bhfext = os.path.splitext(filename)
        bdfext = bhfext
    headername = basename + bhfext
    dataname = basename + bdfext
    bdf = {}
    bdf['his'] = []  # empty list for history
    bdf['C'] = {}  # empty list for bdf file descriptions
    namelists = {}
    valuelists = {}
    with open(headername, 'rb') as fid:  # if fails, an exception is raised
        for line in fid:
            if not line.strip():
                continue  # empty line
            mat = line.split(None, 1)
            prefix = mat[0]
            if prefix == '#C':
                left, right = mat[1].split('=', 1)
                left = left.strip()
                right = right.strip()
                if left in ['xdim', 'ydim']:
                    bdf[left] = int(right)
                elif left in ['type', 'bdf']:
                    bdf[left] = right
                if left in ['Sendtime']:
                    bdf['C'][left] = float(right)
                elif left in ['xdim', 'ydim']:
                    bdf['C'][left] = int(right)
                else:
                    bdf['C'][left] = misc.parse_number(right)
            elif prefix.startswith("#H"):
                bdf['his'].append(mat[1])
#            elif prefix.startswith("#DATA"):
#                if not load_data:
#                    break
#                darray = np.fromfile(fid, dtype = bdf['type'], count = int(bdf['xdim'] * bdf['ydim']))
#                bdf['data'] = np.rot90((darray.reshape(bdf['xdim'], bdf['ydim'])).astype('double').T, 1).copy() # this weird transformation is needed to get the matrix in the same form as bdf_read.m gets it.
#            elif prefix.startswith('#ERROR'):
#                if not load_data:
#                    break
#                darray = np.fromfile(fid, dtype = bdf['type'], count = int(bdf['xdim'] * bdf['ydim']))
#                bdf['error'] = np.rot90((darray.reshape(bdf['xdim'], bdf['ydim'])).astype('double').T, 1).copy()
            else:
                for prf in ['M', 'G', 'S', 'T']:
                    if prefix.startswith('#C%sL' % prf):
                        if prf not in namelists:
                            namelists[prf] = []
                        namelists[prf].extend(mat[1].split())
                    elif prefix.startswith('#C%sV' % prf):
                        if prf not in valuelists:
                            valuelists[prf] = []
                        valuelists[prf].extend([float(x)
                                                for x in mat[1].split()])
                    else:
                        continue
    for dictname, prfname in zip(['M', 'CG', 'CS', 'CT'], ['M', 'G', 'S', 'T']):
        bdf[dictname] = dict(
            list(zip(namelists[prfname], valuelists[prfname])))
    bdf['__Origin__'] = 'BDFv1'
    bdf['__particle__'] = 'photon'

    if load_data:
        f = open(dataname, 'r')
        try:
            s = f.read()
        except IOError as ioe:
            # an ugly bug (M$ KB899149) in W!nd0w$ causes an error if loading too
            # large a file from a network drive and opening it read-only.
            if ioe.errno == 22:
                f.close()
                try:
                    # one work-around is to open it read-write.
                    f = open(dataname, 'r+b')
                    s = f.read()
                except IOError:
                    # if this does not work, inform the user to either obtain
                    # write permission for that file or copy it to a local
                    # drive
                    f.close()
                    raise IOError(22, """
You were probably trying to open a read-only file from a network drive on
Windows, weren\'t you? There is a bug in Windows causing this error
(see http://support.microsoft.com/default.aspx?scid=kb;en-us;899149).
To work around this, please either obtain write permission for that file
(I won't write anything to it, I promise!!!) or copy it to a local drive.
Sorry for the inconvenience.""", ioe.filename)
        datasets = re.findall(
            '#\s*(?P<name>\w+)\[(?P<xsize>\d+):(?P<ysize>\d+)\]', s)
        names = [d[0] for d in datasets]
        xsize = [int(d[1]) for d in datasets]
        ysize = [int(d[2]) for d in datasets]
        dt = np.dtype(bdf['type'])
        for i in range(len(datasets)):
            start = s.find('#%s' % names[i])
            if i < len(datasets) - 1:
                end = s.find('#%s' % (names[i + 1]))
            else:
                end = len(s)
            s1 = s[start:end]
            datasize = xsize[i] * ysize[i] * dt.itemsize
            if datasize > len(s1):
                # assume we are dealing with a BOOL matrix
                bdf[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i]:], dtype=np.uint8)
            else:
                bdf[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i] * dt.itemsize:], dtype=dt)
            # conversion: Matlab saves the array in Fortran-style ordering (columns first).
            # Python however loads in C-style: rows first. We need to take care:
            #   1) reshape from linear to (ysize,xsize) and not (xsize,ysize)
            #   2) transpose (swaps columns and rows)
            # After these operations, we only have to rotate this counter-clockwise by 90
            # degrees because bdf2_write rotates by +270 degrees before saving.
            bdf[names[i]] = np.rot90(
                bdf[names[i]].reshape((ysize[i], xsize[i]), order='F'), 1)

    return bdf


def readbhf(filename, load_data=False, bdfext='.bdf', bhfext='.bhf'):
    if not filename.upper().endswith(bhfext.upper()):
        filename = os.path.splitext(filename)[0] + bhfext
    with open(filename, 'rt') as f:
        version = None
        for l in f:
            m = re.match('#C\sbdfVersion\s*=\s*(\d+\.\d+)', l.strip())
            if m:
                version = float(m.group(1))
                break
    if version is None:
        raise ValueError('Invalid BDF file: no bdfVersion line found!')
    if version < 2:
        return readbhfv1(filename, load_data, bdfext, bhfext)
    else:
        return readbhfv2(filename, load_data, bdfext, bhfext)


def readPAXE(filename, load_data=False):
    f = open(filename, 'r')
    s = f.read()
    f.close()
    par = {}
    par['FSN'] = int(s[2:6])
    par['Owner'] = s[6:12].strip()
    par['Title'] = s[12:0x18].strip()
    par['MeasTime'] = int(s[0x18:0x1e])
    par['Monitor'] = int(s[0x1e:0x26])
    par['Day'] = int(s[0x26:0x28])
    par['Month'] = int(s[0x29:0x2b])
    par['Year'] = int(s[0x2c:0x30])
    par['Hour'] = int(s[0x30:0x32])
    par['Minute'] = int(s[0x33:0x35])
    par['Second'] = int(s[0x36:0x38])
    par['PosSample'] = int(s[0x60:0x65])
    par['PosBS'] = int(s[0x5b:0x60])
    par['PosDetector'] = int(s[0x56:0x5b])
    par['max'] = int(s[0x38:0x3d])
    par['selector_speed'] = int(s[0x3d:0x42])
    par['Wavelength'] = int(s[0x42:0x44])
    par['Dist'] = int(s[0x44:0x49])
    par['comments'] = re.sub(r'\s+', ' ', s[0x6d:0x100].strip())
    try:
        par['sum'] = int(s[0x65:0x6d])
    except:
        par['sum'] = np.nan
    par['BeamPosX'] = float(s[0x49:0x4d])
    par['BeamPosY'] = float(s[0x4d:0x51])
    par['AngleBase'] = float(s[0x51:0x56])
    par['Date'] = datetime.datetime(
        par['Year'], par['Month'], par['Day'], par['Hour'], par['Minute'], par['Second'])
    par['Detector'] = 'XE'
    par['PixelSize'] = 1.0
    par['__Origin__'] = 'PAXE'
    par['__particle__'] = 'neutron'
    if load_data:
        if filename.endswith('32'):
            return par, np.fromstring(s[0x100:], '<i4').astype(np.double).reshape((64, 64))
        else:
            return par, np.fromstring(s[0x100:], '>u2').astype(np.double).reshape((64, 64))
    else:
        return par


def readmarheader(filename):
    """Read a header from a MarResearch .image file."""
    with open(filename, 'rb') as f:
        intheader = np.fromstring(f.read(10 * 4), np.int32)
        floatheader = np.fromstring(f.read(15 * 4), '<f4')
        strheader = f.read(24)
        f.read(4)
        otherstrings = [f.read(16) for i in range(29)]
    return {'Xsize': intheader[0], 'Ysize': intheader[1], 'MeasTime': intheader[8],
            'BeamPosX': floatheader[7], 'BeamPosY': floatheader[8],
            'Wavelength': floatheader[9], 'Dist': floatheader[10],
            '__Origin__': 'MarResearch .image', 'recordlength': intheader[2],
            'highintensitypixels': intheader[4],
            'highintensityrecords': intheader[5],
            'Date': dateutil.parser.parse(strheader),
            'Detector': 'MARCCD', '__particle__': 'photon'}


def readBerSANS(filename):
    """Read a header from a SANS file (produced usually by BerSANS)"""
    hed = {'Comment': ''}
    translate = {'Lambda': 'Wavelength',
                 'Title': 'Owner',
                 'SampleName': 'Title',
                 'BeamcenterX': 'BeamPosY',
                 'BeamcenterY': 'BeamPosX',
                 'Time': 'MeasTime',
                 'TotalTime': 'MeasTime',
                 'Moni1': 'Monitor',
                 'Moni2': 'Monitor',
                 'Moni': 'Monitor',
                 'Transmission': 'Transm',
                 }
    with open(filename, 'rt') as f:
        comment_next = False
        for l in f:
            l = l.strip()
            if comment_next:
                hed['Comment'] = hed['Comment'] + '\n' + l
                comment_next = False
            elif l.startswith('%Counts'):
                break
            elif l.startswith('%Comment'):
                comment_next = True
            elif l.startswith('%'):
                continue
            elif l.split('=', 1)[0] in translate:
                hed[translate[l.split('=', 1)[0]]] = misc.parse_number(
                    l.split('=', 1)[1])
            else:
                try:
                    hed[l.split('=', 1)[0]] = misc.parse_number(
                        l.split('=', 1)[1])
                except IndexError:
                    print(l.split('=', 1))
    if 'FileName' in hed:
        m = re.match('D(\d+)\.(\d+)', hed['FileName'])
        if m is not None:
            hed['FSN'] = int(m.groups()[0])
            hed['suffix'] = int(m.groups()[1])
    if 'FileDate' in hed:
        hed['Date'] = dateutil.parser.parse(hed['FileDate'])
    if 'FileTime' in hed:
        hed['Date'] = datetime.datetime.combine(
            hed['Date'].date(), dateutil.parser.parse(hed['FileTime']).time())
    hed['__Origin__'] = 'BerSANS'
    if 'SD' in hed:
        hed['Dist'] = hed['SD'] * 1000
    if hed['Comment'].startswith('\n'):
        hed['Comment'] = hed['Comment'][1:]
    hed['__particle__'] = 'neutron'
    hed['Wavelength'] *= 10  # convert from nanometres to Angstroems
    return hed
