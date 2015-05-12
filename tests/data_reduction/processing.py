import sastool
import itertools
import collections
import numpy as np
import datetime

datadirs=['/afs/bionano/misc/synchrotron_neutron/desy/hasylab/hasjusi1/ORG/']
sensfsns=list(range(7721,7765))
sensfsns=list(range(7721,7727))
dcfsns=[7817]

#configuration parameters
config={'datadirs':datadirs,
        #energy calibration: dictionary of two lists: apparent and true energies
        'energycalibration':{'apparent':[28468,26092],
                             'literature':[29200,26711]},
        #FSN of dark current
        'darkcurrentFSN':[7817],
        #name of empty beam measurement files
        'emptybeamname':'Empty_beam',
        #the C-style format string of the file names, without extensions and path
        'fileprefixformat':'ORG%05d',
        #If dead time correction is to be done.
        'dodeadtimecorrection':True,
        #Override transmission values with respect to sample names.
        'transmissionoverride':{'Empty_beam':1},
        #Sample-to-detector distances need sometimes to be corrected. Put here
        # the decrements. The default can also be 
        'distancesubtract':collections.defaultdict(lambda:0,{'Reference_on_GC_holder_before_sample_sequence':219,
                                                              'Reference_on_GC_holder_after_sample_sequence':219}),
        #Center-finding. A dictionary, with the following fields:
        #    'mode': 'semitransparent' or 'azimuthal_fold'
        #    'arguments': tuple of additional arguments:
        #          for 'semitransparent', (xmin, xmax, ymin, ymax)
        #          for 'azimuthal_fold',(x0, y0, dmin, dmax, N)
        #    'samplenames': list of samplenames apt for beam finding
        #    'assign': how to assign found coordinates:
        #          'self': determine the beam position from each exposition
        #          In all the other cases, the coordinates are determined from
        #          the samples designated by 'samplenames' and treated:
        #          'nearest': use the coordinates nearest in time
        #          'before': Use the most recently determined beam coordinate
        #          'after': Use the coordinates determined next in time
         
        'centering':{'mode':'azimuthal_fold',
                     'arguments':(128,128,20,80,50),
                     'samplenames':['Reference_on_GC_holder_before_sample_sequence'],
                     'assign':'nearest', #or 'before' or 'after' or 'self'
                     },
        #dictionary of mask file names for each sample.
        'maskfile':collections.defaultdict(lambda:'maskdefault.mat',
                                           {'samplename1':'masksample1.mat'}),
        'masks':{},
        'verbose':True,
       }

    
def config_reloadmasks(config):
    """(Re)load mask files into the config structure"""
    if config['verbose']:
        print("Reloading masks...")
    for k in set(list(config['maskfile'].values()) + [config['maskfile'].default_factory()]):
        config['masks'][k]=sastool.io.twodim.readmask(k,config['datadirs']).astype(np.uint8)
        if config['verbose']:
            print("  reloaded mask",k)
    return config

def config_getmask(config,samplename):
    if config['maskfile'][samplename] not in config['masks']:
        config_reloadmasks(config)
    return config['masks'][config['maskfile'][samplename]]

def summarize(data,dataerr,header):
    print("Summarizing the following samples:")
    for h in header:
        print("  "+describe_header(h))
    datas=sum(data)
    dataerrs=np.sqrt(sum([de**2 for de in dataerr]))
    headers=header[0].copy()
    for k in ['Anode','MeasTime','Monitor','MonitorDORIS','MonitorPIEZO','Transm']:
        headers[k]=sum([h[k] for h in header])
    headers['Transm']/=len(header)
    return datas,dataerrs,headers

def correcttransmission(data,dataerr,header,config):
    if config['verbose']:
        print("Correcting for transmission")
    for i in range(len(header)):
        print("  "+describe_header(header[i]))
        #transmission adjustment
        if header[i]['Title'] in config['transmissionoverride']:
            header[i]['Transm']=config['transmissionoverride'][header[i]['Title']]
            header[i]['TransmError']=0;
        assert 0<header[i]['Transm']<=1
        dataerr[i]=np.sqrt((data[i]/header[i]['Transm']**2*header[i]['TransmError'])**2+\
                           dataerr[i]**2/header[i]['Transm']**2)
        data[i]/=header[i]['Transm']
    return data,dataerr,header

def correctdarkcurrent(data,dataerr,header,config):
    #load dark current measurements
    if ('darkcurrent' not in config) or (config['darkcurrent']['FSN']!=config['darkcurrentFSN']):
        # we must (re)load the dark current measurements 
        dcdata,dcheader=sastool.io.b1.read2dB1data(config['darkcurrentFSN'],
                                                config['fileprefixformat'],config['datadirs'])
        if not dcheader:
            raise IOError('Could not load dark current file!')
        dcerr=[np.sqrt(d) for d in dcdata]
        #summarize dark current files
        dc,dcerr,dcheader=summarize(dcdata,dcerr,dcheader)
        config['darkcurrent']={'data':dc,'error':dcerr,'header':dcheader,
                               'FSN':config['darkcurrentFSN']}
    #dark current correction: subtract time-normalized dark currents.
    data=[d-h['MeasTime']/config['darkcurrent']['header']['MeasTime']*config['darkcurrent']['data'] for d,h in zip(data,header)]
    dataerr=[np.sqrt(de**2+(h['MeasTime']/config['darkcurrent']['header']['MeasTime']*config['darkcurrent']['error'])**2) for \
             de,h in zip(dataerr,header)]
    for i in range(len(header)):
        for k in ['Anode','MeasTime','Monitor','MonitorDORIS','MonitorPIEZO']:
            header[i][k] = header[i][k] - header[i]['MeasTime'] / \
                config['darkcurrent']['header']['MeasTime']*config['darkcurrent']['header'][k]
        header[i][k + "Error"] = np.sqrt(header[i][k + 'Error']**2 + \
            (header[i]['MeasTime'] / config['darkcurrent']['header']['MeasTime'] * \
             config['darkcurrent']['header'][k + 'Error'])**2)
    return data,dataerr,header

def correctdeadtime(data,dataerr,header,config):
    if config['dodeadtimecorrection']:
        dataerr=[np.sqrt(h['AnodeError']**2 *d**2/d.sum()**2  + \
                         h['Anode']*d**2/d.sum()**4*(de**2).sum() + \
                         h['Anode']*de**2*(d.sum()**2-2*d.sum()*d)/d.sum()**4) \
                    for d,de,h in zip(data,dataerr,header)]    
        data=[d*h['Anode']/d.sum() for d,h in zip(data,header)]
    return data,dataerr,header

def correctdistance(data,dataerr,header,config):
    
    for i in range(len(header)):
        #distance calibration
        header[i]['Dist']=header[i]['Dist']-config['distancesubtract'][header[i]['Title']]
    return data,dataerr,header

def correctenergy(data,dataerr,header,config):
    for i in range(len(header)):
        #energy calibration
        header[i]['EnergyCalibrated']=sastool.misc.energycalibration(config['energycalibration']['apparent'],
                                                                     config['energycalibration']['literature'],
                                                                     header[i]['Energy'])
    return data,dataerr,header

def correctmonitor(data,dataerr,header,config):
    if config['verbose']:
        print("Normalizing by monitor counts")
    for i in range(len(header)):
        if config['verbose']:
            print("  "+describe_header(header[i]))
        dataerr[i]=np.sqrt(data[i]**2/header[i]['Monitor']**4*header[i]['MonitorError']**2+\
                           dataerr[i]**2/header[i]['Monitor']**2)
        data[i]=data[i]/header[i]['Monitor']
    return data,dataerr,header

def correctbeamposition(data,dataerr,header,config):
    if config['centering']['assign'].lower()=='self':
        aptsamples=set([h['Title'] for h in header])
    else:
        aptsamples=set(config['centering']['samplenames'])
    # find all beam positions for the 'apt samples'
    beamposfound=[False]*len(header)
    for i in range(len(header)):
        if header[i]['Title'] not in aptsamples: continue
        if config['centering']['mode'].lower()=='semitransparent':
            header[i]['BeamPosX'], header[i]['BeamPosY'] = \
               sastool.utils2d.centering.findbeam_semitransparent(data[i],\
                    config['centering']['arguments'])
        elif config['centering']['mode'].lower()=='azimuthal_fold':
            header[i]['BeamPosX'], header[i]['BeamPosY'] = \
               sastool.utils2d.centering.findbeam_azimuthal_fold(data[i], \
                    config['centering']['arguments'][0:2], \
                    config_getmask(config,header[i]['Title']))
        else:
            raise NotImplementedError('centering mode %s not understood'%config['centering']['mode'])
        beamposfound[i]=True
    # assign beam positions for other samples as well
    for i in range(len(header)):
        if beamposfound[i]: continue
        timedeltas=[(j,header[i]['Date']-h['Date']) for j,bpf,h in \
                       zip(itertools.count(0),beamposfound,header) \
                       if (h is not header[i]) and bpf]
        if config['centering']['assign'].lower()=='self':
            raise NotImplementedError('This should never occur!')
        elif config['centering']['assign'].lower()=='nearest':
            tdmin=min(abs(td[1]) for td in timedeltas)
        elif config['centering']['assign'].lower()=='before':
            tdmin=min(td[1] for td in timedeltas if td[1].total_seconds()>0)
        elif config['centering']['assign'].lower()=='after':
            tdmin=max(td[1] for td in timedeltas if td[1].total_seconds()<0)
        bcidx=[td[0] for td in timedeltas if td[1]==tdmin][0]
        header[i]['BeamPosX'],header[i]['BeamPosY']=(header[bcidx]['BeamPosX'],header[bcidx]['BeamPosY'])
    return data,dataerr,header

def readsequence(fsns,config):
    #load all measurement files
    data,header=sastool.io.b1.read2dB1data(fsns,config['fileprefixformat'],config['datadirs']);
    if not header:
        raise IOError('Could not load any file!')
    dataerr=[np.sqrt(d) for d in data]
    
    
    #we do the dark current correction first, as the anode counter is affected
    # by dark current. On the other hand, the dark current of the scattering
    # image is usually small, therefore no lost counts, no dead time is expected
    
    data,dataerr,header=correctdarkcurrent(data,dataerr,header,config)
    #do dead-time correction if needed
    data,dataerr,header=correctdeadtime(data,dataerr,header,config)
    #do various adjustments
    data,dataerr,header=correctdistance(data,dataerr,header,config)
    data,dataerr,header=correctenergy(data,dataerr,header,config)
    data,dataerr,header=correcttransmission(data,dataerr,header,config)
    data,dataerr,header=correctmonitor(data,dataerr,header,config)
    return data,dataerr,header

def subtractbackground(data,dataerr,header,config):
    pass

def makesensitivity(fsns,fluorescence_energy,config,samplename=None):
    datasens,headersens=sastool.io.b1.read2dB1data(fsns,'ORG%05d.DAT.gz',config['datadirs'])
    if not headersens:
        raise IOError('Could not load any file!')
    if samplename is None:
        samplename=set([h['Title'] for h in headersens if h['Title']!=config['emptybeamname']])[0]
    