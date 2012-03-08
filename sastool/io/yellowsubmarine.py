'''
Created on Feb 15, 2012

@author: andris
'''

from .twodim import readyellowsubmarine
import xlwt
import itertools

def listyellowsubmarine(outputname,fsns,nameformat='XE%04d.DAT',dirs='.'):
    data,header=readyellowsubmarine(nameformat,fsns,dirs)
    wb=xlwt.Workbook()
    ws=wb.add_sheet('Expositions')
    ws.write(0,0,'Run number')
    ws.write(0,1,'Title')
    ws.write(0,2,'Owner')
    ws.write(0,3,'Exposure time, sec')
    ws.write(0,4,'Distance, mm')
    ws.write(0,5,'Comment')
    ws.write(0,6,'Time')
    ws.write(0,7,'Monitor')
    ws.write(0,8,'SamplePosition')
    ws.write(0,9,'DetectorPosition')
    ws.write(0,10,'BeamstopPosition')
    ws.write(0,11,'Sum')
    ws.write(0,12,'Max')
    for i,d,h in zip(itertools.count(1),data,header):
        ws.write(i,0,h['FSN'])
        ws.write(i,1,h['Title'])
        ws.write(i,2,h['Owner'])
        ws.write(i,3,unicode(h['MeasTime']))
        ws.write(i,4,unicode(h['Dist_Ech_det']))
        ws.write(i,5,h['comments'].replace('\n',' ').strip().decode('latin1','replace'))
        ws.write(i,6,unicode(h['Datetime']))
        ws.write(i,7,unicode(h['Monitor']))
        ws.write(i,8,unicode(h['PosSample']))
        ws.write(i,9,unicode(h['PosDetector']))
        ws.write(i,10,unicode(h['PosBS']))
        ws.write(i,11,unicode(d.sum()))
        ws.write(i,12,unicode(d.max()))
    wb.save(outputname)
