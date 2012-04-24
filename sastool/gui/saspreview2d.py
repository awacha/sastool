'''
Created on Feb 14, 2012

@author: andris
'''

import collections
import numpy as np
import itertools
import uuid
import gtk
import gobject
import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.backends
if matplotlib.backends.backend.upper() != 'GTKAGG':
    raise ImportError('Sastool.gui works only with the GTK backend of \
Matplotlib')


from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg, \
NavigationToolbar2GTKAgg
from matplotlib.figure import Figure

from . import patheditor
from ..io import classes
from .. import misc

#Mask matrix should be plotted with plt.imshow(maskmatrix, cmap=_colormap_for_mask)
_colormap_for_mask=matplotlib.colors.ListedColormap(['white','white'],'_sastool_gui_saspreview2d_maskcolormap')
_colormap_for_mask._init()
_colormap_for_mask._lut[:,-1]=0
_colormap_for_mask._lut[0,-1]=0.7

class SAS2DLoader(gtk.VBox):
    _signal_handlers={}
    _previously_opened=None
    def __init__(self,*args,**kwargs):
        gtk.VBox.__init__(self)
        
        b=gtk.Button('Edit search path')
        b.connect('clicked',patheditor.pathedit)
        self.pack_start(b,False)
        
        self.notebook=gtk.Notebook()
        self.pack_start(self.notebook)
        
        ### Input B1 org files
        b1org=gtk.Table()
        self.notebook.append_page(b1org,gtk.Label('B1 org'))
        
        l=gtk.Label('Filename format')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.B1orgfileformat_entry=gtk.Entry()
        self.B1orgfileformat_entry.set_text('org_%05d')
        b1org.attach(self.B1orgfileformat_entry,1,2,0,1,gtk.EXPAND|gtk.FILL,gtk.EXPAND|gtk.FILL)
        
        l=gtk.Label('Image format is autodetected')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,2,1,2,gtk.FILL,gtk.FILL)
        
        l=gtk.Label('FSN')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        self.B1orgfsn_spinbutton=gtk.SpinButton(gtk.Adjustment(0,0,1e10,1,10),digits=0)
        self.B1orgfsn_spinbutton.connect('value-changed',self.on_openfile)
        b1org.attach(self.B1orgfsn_spinbutton,1,2,2,3)
        
        #### Input B1 processed files (int2dnorm*.mat and intnorm*.log)
        
        
        b1proc=gtk.Table()
        self.notebook.append_page(b1proc,gtk.Label('B1 processed'))
        l=gtk.Label('File prefix format:')
        l.set_alignment(0,0.5)
        b1proc.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.B1procfileformat_entry=gtk.Entry()
        self.B1procfileformat_entry.set_text('int2dnorm%d')
        b1proc.attach(self.B1procfileformat_entry,1,2,0,1)
        l=gtk.Label('Logfile format:')
        l.set_alignment(0,0.5)
        b1proc.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.B1proclogformat_entry=gtk.Entry()
        self.B1proclogformat_entry.set_text('intnorm%d.log')
        b1proc.attach(self.B1proclogformat_entry,1,2,1,2)
        l=gtk.Label('FSN')
        l.set_alignment(0,0.5)
        b1proc.attach(l,0,1,2,3)
        self.B1procfsn_spinbutton=gtk.SpinButton(gtk.Adjustment(0,0,1e10,1,10),digits=0)
        self.B1procfsn_spinbutton.connect('value_changed',self.on_openfile)
        b1proc.attach(self.B1procfsn_spinbutton,1,2,2,3)
        
        vb=gtk.VBox()
        self.notebook.append_page(vb,gtk.Label('ID02'))
        
        b=gtk.Button(stock=gtk.STOCK_OPEN)
        self.pack_start(b,False)
        b.connect('clicked',self.on_openfile)
        
    def on_openfile(self,widget):
        if self._previously_opened is None:
            self._previously_opened={'pagename':None,'args':tuple(),'data':None}
        currentpagename=self.notebook.get_tab_label_text(self.notebook.get_nth_page(self.notebook.get_current_page()))
        try:
            if currentpagename=='B1 org':
                fsn=self.B1orgfsn_spinbutton.get_value()
                fileformat=self.B1orgfileformat_entry.get_text()
                dirs=misc.get_search_path()
                if (self._previously_opened['pagename']==currentpagename and 
                    self._previously_opened['args']==(fsn,fileformat,dirs)):
                    print "Not reloading..."
                    return True
                else:
                    data=classes.SASExposure.new_from_B1_org(fsn,fileformat,dirs)
                    self._previously_opened['pagename']=currentpagename
                    self._previously_opened['args']=(fsn,fileformat,dirs)
                    self._previously_opened['data']=data
            elif currentpagename=='B1 processed':
                fsn=self.B1procfsn_spinbutton.get_value()
                fileformat=self.B1procfileformat_entry.get_text()
                logfileformat=self.B1proclogformat_entry.get_text()
                dirs=misc.get_search_path()
                if (self._previously_opened['pagename']==currentpagename and
                    self._previously_opened['args']==(fsn,fileformat,logfileformat,dirs)):
                    print "Not reloading..."
                    return True # do not emit the signal
                else:
                    data = classes.SASExposure.new_from_B1_int2dnorm(fsn,fileformat,logfileformat,dirs)
                    self._previously_opened['pagename']=currentpagename
                    self._previously_opened['args']=(fsn,fileformat,logfileformat,dirs)
                    self._previously_opened['data']=data
            elif currentpagename=='ID02':
                pass
        except IOError:
            return False
        # call the callbacks. This can take some time, this is why we do things so complicated 
        gobject.idle_add(self.call_callbacks)
        return True
    def connect(self,signalname,callbackfunction,*args,**kwargs):
        if signalname=='open-file':
            handler_id=uuid.uuid1()
            self._signal_handlers[handler_id]=(callbackfunction,args,kwargs)
            return handler_id
        else:
            return gtk.VBox.connect(self,signalname,callbackfunction,*args,**kwargs)
    def disconnect(self,handler_id):
        if handler_id in self._signal_handlers:
            del self._signal_handlers[handler_id]
        else:
            return gtk.VBox.disconnect(handler_id)
    def call_callbacks(self):
        gtk.gdk.threads_enter()
        try:
            if self._previously_opened['data'] is not None:
                for func,args,kwargs in self._signal_handlers.itervalues():
                    func(self,self._previously_opened['data'],*args,**kwargs)
        finally:
            gtk.gdk.threads_leave()
        return False  # This is essential to unregister this from the idle callbacks.
class SAS2DPlotter(gtk.VBox):
    data = None
    def __init__(self,figure):
        gtk.VBox.__init__(self)
        self.fig=figure
        table=gtk.Table()
        self.pack_start(table)
        
        l=gtk.Label('Colour scaling')
        l.set_alignment(0,0.5)
        table.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        try:
            self.colourscale=gtk.ComboBoxText()
        except AttributeError:
            self.colourscale=gtk.combo_box_new_text()
        self.colourscale.append_text('Linear')
        self.colourscale.append_text('Nat. logarithmic')
        self.colourscale.append_text('Log10')
        self.colourscale.set_active(0)
        table.attach(self.colourscale,1,2,0,1)

        l=gtk.Label('Matrix to plot')
        l.set_alignment(0,0.5)
        table.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        try:
            self.matrixtype=gtk.ComboBoxText()
        except AttributeError:
            self.matrixtype=gtk.combo_box_new_text()
        for v in classes.SASExposure.matrices.itervalues():
            self.matrixtype.append_text(v)
        self.matrixtype.set_active(0)
        table.attach(self.matrixtype,1,2,1,2)

        
        self.maxint_cb=gtk.CheckButton('Max. value:')
        table.attach(self.maxint_cb,0,1,2,3,gtk.FILL,gtk.FILL)
        self.maxint_entry=gtk.Entry()
        table.attach(self.maxint_entry,1,2,2,3)
        self.maxint_cb.connect('toggled',self.minmaxint_activate,self.maxint_entry)
        self.maxint_cb.set_active(False)
        self.minmaxint_activate(self.maxint_cb,self.maxint_entry)
        
        self.minint_cb=gtk.CheckButton('Min. value:')
        table.attach(self.minint_cb,0,1,3,4,gtk.FILL,gtk.FILL)
        self.minint_entry=gtk.Entry()
        table.attach(self.minint_entry,1,2,3,4)
        self.minint_cb.connect('toggled',self.minmaxint_activate,self.minint_entry)
        self.minint_cb.set_active(False)
        self.minmaxint_activate(self.minint_cb,self.minint_entry)
        
        l=gtk.Label('Colour map')
        l.set_alignment(0,0.5)
        table.attach(l,0,1,4,5,gtk.FILL,gtk.FILL)
        try:
            self.colourmapname=gtk.ComboBoxText()
        except AttributeError:
            self.colourmapname=gtk.combo_box_new_text()
        colourmap_names=[x for x in dir(matplotlib.cm) if eval('isinstance(matplotlib.cm.%s,matplotlib.colors.Colormap)'%x)]
        for v in colourmap_names:
            self.colourmapname.append_text(v)
        self.colourmapname.set_active(colourmap_names.index('jet'))
        table.attach(self.colourmapname,1,2,4,5)
        
        self.plotmask_cb=gtk.CheckButton('Show mask if available')
        table.attach(self.plotmask_cb,0,2,5,6)
        
        hb=gtk.HButtonBox()
        self.pack_start(hb,False,False)
        b=gtk.Button('Replot')
        b.connect('clicked',self.replot)
        hb.add(b)
        b=gtk.Button('Clear axes')
        b.connect('clicked',self.clear)
        hb.add(b)
    def minmaxint_activate(self,cb,entry):
        entry.set_sensitive(cb.get_active());
    def plot2d(self,data):
        if not self.fig.axes:
            return
        valid_attributes=[x for x in data.matrices.keys() if isinstance(data.__getattribute__(x),np.ndarray)]
        if not valid_attributes:
            return
        matrixtype=[x for x in data.matrices.keys() if data.matrices[x]==self.matrixtype.get_active_text()][0]
        
        if matrixtype not in valid_attributes:
            matrixtype=valid_attributes[0]
            self.matrixtype.set_active([i for i,k in zip(itertools.count(0),data.matrices.keys()) if k==matrixtype][0])
        mat=data.__getattribute__(matrixtype).copy()
        if self.minint_cb.get_active():
            minint=float(self.minint_entry.get_text())
        else:
            minint=-np.inf
            self.minint_entry.set_text(str(np.nanmin(mat)))
        if self.maxint_cb.get_active():
            maxint=float(self.maxint_entry.get_text())
        else:
            maxint=np.inf
            self.maxint_entry.set_text(str(np.nanmax(mat)))
        if minint>=maxint:
            raise ValueError('Min. value is larger than max. value')
        mat[mat<minint]=np.nanmin(mat[mat>=minint])
        mat[mat>maxint]=np.nanmax(mat[mat<=maxint])
        if self.colourscale.get_active_text()=='Log10':
            mat=np.log10(mat)
        elif self.colourscale.get_active_text()=='Nat. logarithmic':
            mat=np.log(mat)
        elif self.colourscale.get_active_text()=='Linear':
            mat=mat
        else:
            raise NotImplementedError(self.colourscale.get_active_text())
        self.fig.axes[0].cla()
        self.fig.axes[0].imshow(mat,cmap=eval('matplotlib.cm.%s'%self.colourmapname.get_active_text()),
                                interpolation='nearest')
        if self.plotmask_cb.get_active() and data.mask is not None:
            self.fig.axes[0].imshow(mat,cmap=_colormap_for_mask,interpolation='nearest')
        self.fig.axes[0].set_title(str(data.header))
        self.fig.axes[0].set_axis_bgcolor('black')
        self.fig.canvas.draw()
        if self.data is not None:
            del self.data
        self.data=data
    def replot(self,widget=None):
        if self.data is not None:
            self.plot2d(self.data)
    def clear(self,widget=None):
        if self.fig.axes:
            self.fig.axes[0].cla()
            self.fig.axes[0].set_axis_bgcolor('white')
            self.fig.canvas.draw()

class SAS2DMasker(gtk.VBox):
    def __init__(self):
        gtk.VBox.__init__(self)
        
    def getmask(self):
        pass

class SAS2DStatistics(gtk.Frame):
    def __init__(self):
        gtk.Frame.__init__(self,'Statistics')
        self.table=gtk.Table()
        self.table_widgets=[]
        self.add(self.table)
    def clear_table(self):
        for tw in self.table_widgets:
            self.table.remove(tw)
    def update_table(self,statistics,clear=True):
        if clear:
            self.clear_table()
        for k,i in zip(statistics.keys(),itertools.count(0)):
            self.table_widgets.append(gtk.Label(k+':'))
            self.table_widgets[-1].set_alignment(0,0.5)
            self.table.attach(self.table_widgets[-1],0,1,i,i+1,gtk.FILL,gtk.FILL)
            self.table_widgets.append(gtk.Label(statistics[k]))
            self.table_widgets[-1].set_alignment(0,0.5)
            self.table.attach(self.table_widgets[-1],1,2,i,i+1)
        self.table.show_all()

class SAS2DGUI(gtk.Window):
    data=None
    def __init__(self,*args,**kwargs):
        gtk.Window.__init__(self,*args,**kwargs)
        self.set_title('SAS 2D GUI tool')
        
        hb=gtk.HBox()
        self.add(hb)
        vb=gtk.VBox()
        hb.pack_start(vb,False,False)
        
        # add a matplotlib figure
        figvbox = gtk.VBox()
        hb.pack_start(figvbox)
        self.fig = Figure(figsize = (0.5, 0.4), dpi = 72)
        self.axes=self.fig.add_subplot(111)

        self.canvas = FigureCanvasGTKAgg(self.fig)
        self.canvas.set_size_request(300, 200)
        figvbox.pack_start(self.canvas, True, True, 0)
        #self.canvas.mpl_connect('button_press_event', self._on_matplotlib_mouseclick)

        hb1 = gtk.HBox() # the toolbar below the figure
        figvbox.pack_start(hb1, False, True, 0)
        self.graphtoolbar = NavigationToolbar2GTKAgg(self.canvas, hb)
        hb1.pack_start(self.graphtoolbar, True, True, 0)
        
        #build the toolbar on the left pane
        ex=gtk.Expander(label='Load measurement')
        vb.pack_start(ex,False,True)
        self.loader=SAS2DLoader()
        self.loader.connect('open-file',self.file_opened)
        ex.add(self.loader)
        ex.set_expanded(True)
        
        ex=gtk.Expander(label='Plotting')
        vb.pack_start(ex,False,True)
        ex.set_expanded(True)
        self.plotter=SAS2DPlotter(self.fig)
        ex.add(self.plotter)
        
        ex=gtk.Expander(label='Masking')
        vb.pack_start(ex,False,True)
        self.masker=SAS2DMasker()
        ex.add(self.masker)
        
        integrator=gtk.Expander(label='Radial averaging')
        vb.pack_start(integrator,False,True)
        centerer=gtk.Expander(label='Centering')
        vb.pack_start(centerer,False,True)
        
        self.statistics=SAS2DStatistics()
        vb.pack_end(self.statistics,False)
        
        self.show_all()
        self.hide()
    def file_opened(self,widget,exposition):
        if self.data is not None:
            del self.data
        self.data=exposition
        self.plotter.plot2d(exposition)
        matrixtype=[x for x in exposition.matrices.keys() if exposition.matrices[x]==self.plotter.matrixtype.get_active_text()][0]
        sumdata=np.nansum(exposition.__getattribute__(matrixtype))
        Nandata=(-np.isfinite(exposition.__getattribute__(matrixtype))).sum()
        maxdata=np.nanmax(exposition.__getattribute__(matrixtype))
        mindata=np.nanmin(exposition.__getattribute__(matrixtype))
        self.statistics.update_table(collections.OrderedDict([('FSN',str(self.data.header['FSN'])),
                                                              ('Title',self.data.header['Title']),
                                                              ('Total counts',sumdata),
                                                              ('Invalid pixels',Nandata),
                                                              ('Max. count',maxdata),
                                                              ('Min. count',mindata)]))

def SAS2DGUI_run():
    w = SAS2DGUI()
    w.show()
    