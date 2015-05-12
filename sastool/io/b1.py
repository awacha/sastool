import numbers
import h5py

from ..misc import findfileindirs, normalize_listargument
from ..classes import SASHeader, SASExposure, SASMask, SASCurve


def read2dB1data(fsns, fileformat = 'org_%05d', dirs = []):
    """Read 2D measurement files, along with their header data

    Inputs:
        fsns: the file sequence number or a list (or tuple or set or np.ndarray 
            of them).
        fileformat: format of the file name (without the extension!)
        dirs [optional]: a list of directories to try
        
    Outputs: datas, headers
        datas: A list of 2d scattering data matrices (numpy arrays)
        headers: A list of header data (Python dicts)
        
    Examples:
        Read FSN 123-130:
        a) measurements with the Gabriel detector:
        data,header=read2dB1data(range(123,131),'ORG%05d')
        b) measurements with a Pilatus* detector:
        #in this case the org_*.header files should be present
        data,header=read2dB1data(range(123,131),'org_%05d')
    """
    fsns = normalize_listargument(fsns)
    datas = []
    headers = []
    for f in fsns:
        try:
            data = SASExposure.new_from_B1_org(f, fileformat, dirs)
        except IOError:
            continue #skip this file
        datas.append(data)
        headers.append(data.header)
    return datas, headers

def read2dintfile(fsn, fileformat = 'int2dnorm%d', logfileformat = 'intnorm%d.log', dirs = []):
    fsns = normalize_listargument(fsn)
    def read_and_eat_exception(f):
        try:
            return SASExposure.new_from_B1_int2dnorm(f, fileformat, logfileformat, dirs)
        except IOError:
            print("Could not load files for FSN", f)
            return None
    loaded = [read_and_eat_exception(f) for f in fsns]
    loaded = [l for l in loaded if l is not None]
    if isinstance(fsn, numbers.Number) and loaded:
        return loaded[0]
    return loaded

def readparamfile(filename):
    """Read param files (*.log)
    
    Inputs:
        filename: the file name

    Output: the parameter dictionary
    """
    return SASHeader.new_from_B1_log(filename)

def readlogfile(fsns, paramformat = 'intnorm%d.log', dirs = []):
    fsns = normalize_listargument(fsns)
    logfiles = []
    for f in fsns:
        try:
            logfiles.append(readparamfile(findfileindirs(paramformat % f, dirs)))
        except IOError:
            pass
    return logfiles

def writeparamfile(filename, param):
    """Write the param structure into a logfile. See writelogfile() for an explanation.
    
    Inputs:
        filename: name of the file.
        param: param structure (dictionary).
        
    Notes:
        all exceptions pass through to the caller.
    """
    return SASHeader(param).write_B1_log(filename)

def read1d(fsns, fileformat = 'intnorm%d.dat', paramformat = 'intnorm%d.log', dirs = []):
    fsns = normalize_listargument(fsns)
    datas = []
    params = []
    for f in fsns:
        try:
            filename = findfileindirs(fileformat % f, dirs)
            paramname = findfileindirs(paramformat % f, dirs)
        except IOError:
            continue
        data = SASCurve.new_from_file(filename)
        param = readparamfile(paramname)
        data.header = param
        datas.append(data)
        params.append(param)
    return datas, params

def readbinned(fsns, *args, **kwargs):
    return read1d(fsns, *args, fileformat = 'intbinned%d.dat', **kwargs)

def readsummed(fsns, *args, **kwargs):
    return read1d(fsns, *args, fileformat = 'summed%d.dat', paramformat = 'summed%d.log', **kwargs)

def readunited(fsns, *args, **kwargs):
    return read1d(fsns, *args, fileformat = 'united%d.dat', paramformat = 'united%d.log', **kwargs)

def readwaxscor(fsns, *args, **kwargs):
    return read1d(fsns, *args, fileformat = 'waxs%d.cor', **kwargs)




def convert_B1intnorm_to_HDF5(fsns, hdf5_filename, maskrules,
                              int2dnormformat = 'int2dnorm%d',
                              logformat = 'intnorm%d.log', dirs = []):
    """Convert int2dnorm.mat files to hdf5 format.
    
    Inputs:
        fsns: list of file sequence numbers
        hdf5_filename: name of HDF5 file
        maskrules: mask definition rules. It is a list of tuples, each tuple
            corresponding to a mask rule:
                (mask_name_or_instance, rule_list)
            where mask_name_or_instance is a string mask name (e.g. mask4 for
            mask4.mat) or a SASMask instance. rule_list is a dictionary, its
            keys being valid header keys (Dist, Energy, Title etc) and the
            values are either: 1) callables. In this case matching is determined
            by calling the function with the corresponding header value as its
            sole argument, or 2) simple values, comparision is made with '=='.
        int2dnormformat: C-style (.mat and .npz will be tried)!
        logformat: C-style file format for the log files, with extension.
        dirs: search path definition
        
    Outputs:
        None, the HDF5 file in hdf5_filename will be written.
    """
    class MatchesException(Exception):
        pass
    def mask_rule_matches(headerval, value_or_function):
        if hasattr(value_or_function, '__call__'):
            return value_or_function(headerval)
        else:
            return value_or_function == headerval
    fsns = normalize_listargument(fsns)
    hdf = h5py.highlevel.File(hdf5_filename)
    for f in fsns:
        # read the exposition
        try:
            a = SASExposure.new_from_B1_int2dnorm(f, int2dnormformat, logformat, dirs)
        except IOError:
            print("Could not open file", f)
            continue
        try:
            # find a matching mask
            for m, rulesdict in maskrules:
                if all([mask_rule_matches(a.header[k], rulesdict[k]) for k in rulesdict]):
                    raise MatchesException(m)
        except MatchesException as me:
            mask = me.args[0]
        else:
            raise ValueError('No mask found for ' + str(a.header))
        if isinstance(mask, str):
            mask = findfileindirs(mask, dirs)
        mask = SASMask(mask)
        a.set_mask(mask)
        a.write_to_hdf5(hdf)
        del mask
        del a
        print("Converted ", f)
    hdf.close()
