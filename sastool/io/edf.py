#Input-output routines for EDF files (ESRF ID01 and ID02
# among others)
import dateutil.parser
import re
import glob
import xlwt
import itertools
from twodim import readehf


def listedf(outputname,fileglob='*/*ccd'):
    filenames=glob.glob(fileglob)
    wb=xlwt.Workbook()
    ws=wb.add_sheet('Expositions')
    ws.write(0,0,'File name')
    ws.write(0,1,'Title')
    ws.write(0,2,'Transmission')
    ws.write(0,3,'Exposure time, sec')
    ws.write(0,4,'Distance, mm')
    ws.write(0,5,'Time')
    ws.write(0,6,'Wavelength, nm')
    ws.write(0,7,'Motor SFD')
    ws.write(0,8,'Motor SAmple X')
    ws.write(0,9,'Motor SAmple Z')
    ws.write(0,10,'Motor TABLE Z')
    ws.write(0,11,'Binning')
    ws.write(0,12,'Pixel size, mm')
    ws.write(0,13,'Mask file')
    for i,f in zip(itertools.count(1),sorted(filenames)):
        if not i%100:
            print "%d/%d"%(i,len(filenames))
        edf=readehf(f)
        ws.write(i,0,f)
        ws.write(i,1,edf['TitleBody'])
        ws.write(i,2,unicode(edf['Intensity1']/edf['Intensity0']))
        ws.write(i,3,unicode(edf['ExposureTime']))
        ws.write(i,4,unicode(edf['SampleDistance']*1e3))
        ws.write(i,5,unicode(edf['Time']))
        ws.write(i,6,unicode(edf['WaveLength']*1e11))
        ws.write(i,7,unicode(edf['ESRF_ID2_SAXS_SFD']))
        ws.write(i,8,unicode(edf['ESRF_ID2_SAXS_SAX']))
        ws.write(i,9,unicode(edf['ESRF_ID2_SAXS_SAZ']))
        ws.write(i,10,unicode(edf['ESRF_ID2_SAXS_TABLEZ']))
        ws.write(i,11,'%d x %d'%(edf['BSize_1'],edf['BSize_2']))
        ws.write(i,12,'%.3f x %.3f'%(edf['PSize_1']*1e3,edf['PSize_2']*1e3))
        ws.write(i,13,unicode(edf['MaskFileName']))
    wb.save(outputname)
