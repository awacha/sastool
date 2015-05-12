'''
Created on Jun 13, 2012

@author: andris
'''

import xlwt
import glob
import numpy as np
import os

from .twodim import readPAXE
from ..misc import findfileindirs
from .header import readehf, readB1header
import itertools
from .onedim import readabt

def listyellowsubmarine(outputname, fsns, nameformat = 'XE%04d.DAT', dirs = '.'):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Expositions')
    ws.write(0, 0, 'Run number')
    ws.write(0, 1, 'Title')
    ws.write(0, 2, 'Owner')
    ws.write(0, 3, 'Exposure time, sec')
    ws.write(0, 4, 'Distance, mm')
    ws.write(0, 5, 'Comment')
    ws.write(0, 6, 'Time')
    ws.write(0, 7, 'Monitor')
    ws.write(0, 8, 'SamplePosition')
    ws.write(0, 9, 'DetectorPosition')
    ws.write(0, 10, 'BeamstopPosition')
    ws.write(0, 11, 'Sum')
    ws.write(0, 12, 'Max')
    i = 1
    for f in fsns:
        try:
            h, d = readPAXE(findfileindirs(nameformat % f, dirs))
        except IOError:
            continue
        print(f)
        ws.write(i, 0, h['FSN'])
        ws.write(i, 1, h['Title'].decode('latin1', 'replace'))
        ws.write(i, 2, h['Owner'].decode('latin1', 'replace'))
        ws.write(i, 3, h['MeasTime'])
        ws.write(i, 4, h['Dist'])
        ws.write(i, 5, h['comments'].replace('\n', ' ').strip().decode('latin1', 'replace'))
        ws.write(i, 6, str(h['Date']))
        ws.write(i, 7, h['Monitor'])
        ws.write(i, 8, h['PosSample'])
        ws.write(i, 9, h['PosDetector'])
        ws.write(i, 10, h['PosBS'])
        ws.write(i, 11, d.sum())
        ws.write(i, 12, d.max())
        del d, h
        i += 1
    wb.save(outputname)

def listedf(outputname, fileglob = '*/*ccd'):
    filenames = glob.glob(fileglob)
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Expositions')
    ws.write(0, 0, 'File name')
    ws.write(0, 1, 'Title')
    ws.write(0, 2, 'Transmission')
    ws.write(0, 3, 'Exposure time, sec')
    ws.write(0, 4, 'Distance, mm')
    ws.write(0, 5, 'Time')
    ws.write(0, 6, 'Wavelength, nm')
    ws.write(0, 7, 'Motor SFD')
    ws.write(0, 8, 'Motor SAmple X')
    ws.write(0, 9, 'Motor SAmple Z')
    ws.write(0, 10, 'Motor TABLE Z')
    ws.write(0, 11, 'Binning')
    ws.write(0, 12, 'Pixel size, mm')
    ws.write(0, 13, 'Mask file')
    for i, f in zip(itertools.count(1), sorted(filenames)):
        edf = readehf(f)
        ws.write(i, 0, f)
        ws.write(i, 1, edf['TitleBody'])
        ws.write(i, 2, edf['Intensity1'] / edf['Intensity0'])
        ws.write(i, 3, edf['ExposureTime'])
        ws.write(i, 4, edf['SampleDistance'] * 1e3)
        ws.write(i, 5, str(edf['Time']))
        ws.write(i, 6, edf['WaveLength'] * 1e11)
        ws.write(i, 7, edf['ESRF_ID2_SAXS_SFD'])
        ws.write(i, 8, edf['ESRF_ID2_SAXS_SAX'])
        ws.write(i, 9, edf['ESRF_ID2_SAXS_SAZ'])
        ws.write(i, 10, edf['ESRF_ID2_SAXS_TABLEZ'])
        ws.write(i, 11, '%d x %d' % (edf['BSize_1'], edf['BSize_2']))
        ws.write(i, 12, '%.3f x %.3f' % (edf['PSize_1'] * 1e3, edf['PSize_2'] * 1e3))
        ws.write(i, 13, str(edf['MaskFileName']))
    wb.save(outputname)

def listabtfiles(directory = '.', fileformat = 'abt*.fio'):
    lis = glob.glob(os.path.join(directory, fileformat))
    for filename in sorted(lis):
        try:
            abt = readabt(filename, [''])
        except IOError:
            pass
        print(abt['name'], abt['scantype'], abt['title'], abt['start'].isoformat(), abt['end'].isoformat())

def listB1(fsns, xlsname, dirs, whattolist = None, headerformat = 'org_%05d.header'):
    """ getsamplenames revisited, XLS output.
    
    Inputs:
        fsns: FSN sequence
        xlsname: XLS file name to output listing
        dirs: either a single directory (string) or a list of directories, a la readheader()
        whattolist: format specifier for listing. Should be a list of tuples. Each tuple
            corresponds to a column in the worksheet, in sequence. The first element of
            each tuple is the column title, eg. 'Distance' or 'Calibrated energy (eV)'.
            The second element is either the corresponding field in the header dictionary
            ('Dist' or 'EnergyCalibrated'), or a tuple of them, eg. ('FSN', 'Title', 'Energy').
            If the column-descriptor tuple does not have a third element, the string
            representation of each field (str(param[i][fieldname])) will be written
            in the corresponding cell. If a third element is present, it is treated as a 
            format string, and the values of the fields are substituted.
        headerformat: C-style format string of header file names (e.g. org_%05d.header)
        
    Outputs:
        an XLS workbook is saved.
    
    Notes:
        if whattolist is not specified exactly (ie. is None), then the output
            is similar to getsamplenames().
        module xlwt is needed in order for this function to work. If it cannot
            be imported, the other functions may work, only this function will
            raise a NotImplementedError.
    """
    if whattolist is None:
        whattolist = [('FSN', 'FSN'), ('Time', 'MeasTime'), ('Energy', 'Energy'),
                    ('Distance', 'Dist'), ('Position', 'PosSample'),
                    ('Transmission', 'Transm'), ('Temperature', 'Temperature'),
                    ('Title', 'Title'), ('Date', ('Day', 'Month', 'Year', 'Hour', 'Minutes'), '%02d.%02d.%04d %02d:%02d')]
    wb = xlwt.Workbook(encoding = 'utf8')
    ws = wb.add_sheet('Measurements')
    for i in range(len(whattolist)):
        ws.write(0, i, whattolist[i][0])
    i = 1
    for fsn in fsns:
        try:
            hed = readB1header(findfileindirs(headerformat % fsn, dirs))
        except IOError:
            continue
        # for each param structure create a line in the table
        for j in range(len(whattolist)):
            # for each parameter to be listed, create a column
            if np.isscalar(whattolist[j][1]):
                # if the parameter is a scalar, make it a list
                fields = tuple([whattolist[j][1]])
            else:
                fields = whattolist[j][1]
            if len(whattolist[j]) == 2:
                if len(fields) >= 2:
                    strtowrite = ''.join([str(hed[f]) for f in fields])
                else:
                    strtowrite = hed[fields[0]]
            elif len(whattolist[j]) >= 3:
                strtowrite = whattolist[j][2] % tuple([hed[f] for f in fields])
            else:
                assert False
            ws.write(i, j, strtowrite)
        i += 1
    wb.save(xlsname)
