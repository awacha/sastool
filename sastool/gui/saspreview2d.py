'''
Created on Feb 14, 2012

@author: andris
'''

import re
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
from . import maskmaker
from .. import utils2d

class sastool_break_loop(Exception):
    pass

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
        
        #### Input measurements from ID02, ESRF
        
        id2=gtk.Table()
        self.notebook.append_page(id2,gtk.Label('ID02'))
        l=gtk.Label('File format:')
        l.set_alignment(0,0.5)
        id2.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.ID2fileformat_entry=gtk.Entry()
        self.ID2fileformat_entry.set_text('xxyyyy_0_%04dccd')
        id2.attach(self.ID2fileformat_entry,1,2,0,1)
        l=gtk.Label('FSN')
        l.set_alignment(0,0.5)
        id2.attach(l,0,1,1,2)
        self.ID2fsn_spinbutton=gtk.SpinButton(gtk.Adjustment(0,0,1e10,1,10),digits=0)
        self.ID2fsn_spinbutton.connect('value_changed',self.on_openfile)
        id2.attach(self.ID2fsn_spinbutton,1,2,1,2)
        self.ID2create_errormatrix_cb=gtk.CheckButton("Estimate errors")
        self.ID2create_errormatrix_cb.set_active(True)
        id2.attach(self.ID2create_errormatrix_cb,0,2,2,3)
        
        b=gtk.Button(stock=gtk.STOCK_OPEN)
        self.pack_start(b,False)
        b.connect('clicked',self.on_openfile)
        
    def on_openfile(self,widget):
        if self._previously_opened is None:
            self._previously_opened={'pagename':None,'args':tuple(),'data':None}
        currentpagename=self.notebook.get_tab_label_text(self.notebook.get_nth_page(self.notebook.get_current_page()))
        dirs=misc.get_search_path()
        try:
            if currentpagename=='B1 org':
                fsn=self.B1orgfsn_spinbutton.get_value()
                fileformat=self.B1orgfileformat_entry.get_text()
                args=(fsn,fileformat,dirs)
                func=classes.SASExposure.new_from_B1_org
            elif currentpagename=='B1 processed':
                fsn=self.B1procfsn_spinbutton.get_value()
                fileformat=self.B1procfileformat_entry.get_text()
                logfileformat=self.B1proclogformat_entry.get_text()
                args=(fsn,fileformat,logfileformat,dirs)
                func=classes.SASExposure.new_from_B1_int2dnorm
            elif currentpagename=='ID02':
                fsn=self.ID2fsn_spinbutton.get_value()
                fileformat=self.ID2fileformat_entry.get_text()
                estimate_errors=self.ID2create_errormatrix_cb.get_active()
                args=(fsn,fileformat,estimate_errors,dirs)
                func=classes.SASExposure.new_from_ESRF_ID02
            else:
                raise NotImplementedError(currentpagename)
            if (self._previously_opened['pagename']==currentpagename and
                self._previously_opened['args']==args):
                print "Not reloading..."
                return True
            else:
                data=func(*args)
                self._previously_opened['pagename']=currentpagename
                self._previously_opened['args']=args
                self._previously_opened['data']=data
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
    def __init__(self,figure):
        data = None
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
        self.colourscale.connect('changed',self.replot)
        
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
        self.matrixtype.connect('changed',self.replot)
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
        self.colourmapname.connect('changed',self.replot)
        table.attach(self.colourmapname,1,2,4,5)
        
        self.plotmask_cb=gtk.CheckButton('Show mask if available')
        self.plotmask_cb.set_active(True)
        self.plotmask_cb.connect('toggled',self.replot)
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
    def update_matrixtype(self,data,name=None):
        if name is None:
            name=self.matrixtype.get_active_text()
        matrixtype=data.get_matrix_name(name)
        self.matrixtype.set_active(data.matrices.keys().index(matrixtype))
    def get_matrixtype(self):
        return [k for k in classes.SASExposure.matrices if classes.SASExposure.matrices[k]==self.matrixtype.get_active_text()][0]
    def plot2d(self,data):
        if hasattr(self,'mat'):
            del self.mat
        if hasattr(self,'data'):
            del self.data
            self.data=None
        if not self.fig.axes:
            return
        self.update_matrixtype(data)
        mat=data.get_matrix(self.matrixtype.get_active_text())
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
        self.cmap=eval('matplotlib.cm.%s'%self.colourmapname.get_active_text())
        self.fig.axes[0].imshow(mat,cmap=self.cmap,
                                interpolation='nearest')
        if self.plotmask_cb.get_active() and data.mask is not None:
            self.fig.axes[0].imshow(data.mask.mask,cmap=_colormap_for_mask,interpolation='nearest')
        else:
            self.plotmask_cb.get_active()
        self.fig.axes[0].set_title(str(data.header))
        self.fig.axes[0].set_axis_bgcolor('black')
        
        if len(self.fig.axes)>1:
            self.fig.colorbar(self.fig.axes[0].images[0],cax=self.fig.axes[1])
        else:
            self.fig.colorbar(self.fig.axes[0].images[0])
        
        self.fig.canvas.draw()
        self.data=data
        self.mat=mat
    def replot(self,widget=None):
        if not hasattr(self,'data'):
            return
        if self.data is not None:
            self.plot2d(self.data)
    def clear(self,widget=None):
        self.fig.clf()
        self.fig.add_subplot(111)
        self.fig.axes[0].set_axis_bgcolor('white')
        self.fig.canvas.draw()

class SAS2DMasker(gtk.VBox):
    mask=None
    def __init__(self,matrix_source=None):
        gtk.VBox.__init__(self)
        self.matrix_source=matrix_source
        tab=gtk.Table()
        self.pack_start(tab,False)
        l=gtk.Label('Loaded mask:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.maskid_entry=gtk.Entry()
        self.maskid_entry.set_text('None so far')
        self.maskid_entry.set_sensitive(False)
        self.maskid_entry.connect('changed',self.maskid_changed)
        tab.attach(self.maskid_entry,1,2,0,1)
        
        l=gtk.Label('Mask shape:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.shape_label=gtk.Label('None so far')
        self.shape_label.set_alignment(0,0.5)
        tab.attach(self.shape_label,1,2,1,2)
        
        hb=gtk.HBox()
        self.pack_start(hb,False)
        b=gtk.Button('Attach to dataset')
        b.connect('clicked',self.updatemaskindata)
        hb.pack_start(b)
        self.autoupdate_cb=gtk.CheckButton('Auto-attach')
        self.autoupdate_cb.set_active(True)
        hb.pack_start(self.autoupdate_cb)
        
        bb=gtk.HButtonBox()
        self.pack_start(bb,False)
        b=gtk.Button(stock=gtk.STOCK_OPEN)
        b.connect('clicked',self.openmask)
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_EDIT)
        b.connect('clicked',self.editmask)
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_SAVE)
        b.connect('clicked',self.savemask)
        bb.add(b)
    def maskid_changed(self,widget):
        if self.mask is not None:
            self.mask.maskid=widget.get_text()
        return False
    def setmask(self,mask):
        self.mask=mask
        self.maskid_entry.set_text(self.mask.maskid)
        self.maskid_entry.set_sensitive(True)
        self.shape_label.set_text('%d x %d'%self.mask.mask.shape)
    def getmask(self):
        return self.mask
    def openmask(self,widget):
        if not hasattr(self,'open_fcd'):
            self.open_fcd=gtk.FileChooserDialog('Open mask file...',self.get_toplevel(),
                                                gtk.FILE_CHOOSER_ACTION_OPEN,
                                                (gtk.STOCK_OK,gtk.RESPONSE_OK,
                                                 gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL))
            ff=gtk.FileFilter()
            ff.set_name('All mask files')
            ff.add_pattern('*.mat')
            ff.add_pattern('*.npy')
            ff.add_pattern('*.npz')
            ff.add_pattern('*.edf')
            self.open_fcd.add_filter(ff)
            ff=gtk.FileFilter()
            ff.set_name('All files')
            ff.add_pattern('*')
            self.open_fcd.add_filter(ff)
            ff=gtk.FileFilter()
            ff.set_name('Matlab (R) matrices')
            ff.add_pattern('*.mat')
            self.open_fcd.add_filter(ff)
            ff=gtk.FileFilter()
            ff.set_name('Numpy arrays')
            ff.add_pattern('*.npy')
            ff.add_pattern('*.npz')
            self.open_fcd.add_filter(ff)
            ff=gtk.FileFilter()
            ff.set_name('ESRF Data Format')
            ff.add_pattern('*.edf')
            self.open_fcd.add_filter(ff)
        if self.open_fcd.run()==gtk.RESPONSE_OK:
            self.setmask(classes.SASMask(self.open_fcd.get_filename()))
        self.updatemaskindata()
        self.open_fcd.hide()
    def savemask(self,widget):
        pass
    def editmask(self,widget):
        mm=maskmaker.MaskMaker(matrix=self.matrix_source.get_matrix(),mask=self.mask.mask.copy())
        if mm.run()==gtk.RESPONSE_OK:
            mask=classes.SASMask(mm.get_mask())
            if re.match('^.*_\d+$',self.mask.maskid) is None:
                mask.maskid=self.mask.maskid+'_1'
            else:
                l,r=self.mask.maskid.rsplit('_',1)
                mask.maskid=l+'_'+str(int(r)+1)
            self.setmask(mask)
        mm.destroy()
        self.updatemaskindata(True)
    def updatemaskindata(self,widget_or_force=None):
        # this will update the mask in the currently loaded dataset, if:
        #  1) widget_or_force is None and the auto-update checkbutton is checked
        #  2) widget_or_force is not None
        if widget_or_force is None and not self.autoupdate_cb.get_active():
            return True
        if self.mask is None:
            return True
        if self.matrix_source.get_matrix() is None:
            return True
        datashape=self.matrix_source.get_matrix().shape
        maskshape=self.mask.mask.shape
        if datashape!=maskshape:
            d=gtk.MessageDialog(self.get_toplevel(),
                                gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,
                                gtk.MESSAGE_ERROR, gtk.BUTTONS_OK,
                                'Mask shape %s is incompatible with data shape %s'%(maskshape,datashape))
            d.run()
            d.destroy()
        else:
            self.matrix_source.get_data().set_mask(self.mask)
            self.matrix_source.refresh_stats()
            self.matrix_source.replot()
        return True
        

class SAS2DStatistics(gtk.Frame):
    def __init__(self):
        gtk.Frame.__init__(self,'Statistics')
        sw=gtk.ScrolledWindow()
        self.add(sw)
        sw.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.table=gtk.Table()
        self.table_widgets=[]
        sw.add_with_viewport(self.table)
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

class SAS2DCenterer(gtk.VBox):
    def __init__(self,matrix_source=None):
        super(SAS2DCenterer,self).__init__()
        self.matrix_source=matrix_source
        self.notebook=gtk.Notebook()
        self.pack_start(self.notebook)
        #self.notebook.set_tab_pos(gtk.POS_LEFT)
        self.notebook.set_scrollable(True)
        ### semitransparent beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Semitransp.'))
        l=gtk.Label('row min:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.pri_rmin_entry=gtk.Entry()
        self.pri_rmin_entry.set_text('0')
        tab.attach(self.pri_rmin_entry,1,2,0,1)

        l=gtk.Label('row max:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.pri_rmax_entry=gtk.Entry()
        self.pri_rmax_entry.set_text('1')
        tab.attach(self.pri_rmax_entry,1,2,1,2)

        l=gtk.Label('col min:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        self.pri_cmin_entry=gtk.Entry()
        self.pri_cmin_entry.set_text('0')
        tab.attach(self.pri_cmin_entry,1,2,2,3)

        l=gtk.Label('col max:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,3,4,gtk.FILL,gtk.FILL)
        self.pri_cmax_entry=gtk.Entry()
        self.pri_cmax_entry.set_text('1')
        tab.attach(self.pri_cmax_entry,1,2,3,4)
        
        b=gtk.Button('Get from current zoom')
        b.connect('clicked',self.get_pri_from_zoom)
        tab.attach(b,0,2,4,5)
        ### by-hand beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Click'))
        l=gtk.Label('This algorithm has\nno parameters')
        l.set_justify(gtk.JUSTIFY_CENTER)
        l.set_alignment(0.5,0.5)
        tab.attach(l,0,1,0,1)
        ### gravity beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Gravity'))
        l=gtk.Label('This algorithm has\nno parameters')
        l.set_justify(gtk.JUSTIFY_CENTER)
        l.set_alignment(0.5,0.5)
        tab.attach(l,0,1,0,1)
        
        ### sectors beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Sectors'))
        l=gtk.Label('Min. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.sector_dmin_entry=gtk.Entry()
        self.sector_dmin_entry.set_text('0')
        tab.attach(self.sector_dmin_entry,1,2,0,1)

        l=gtk.Label('Max. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.sector_dmax_entry=gtk.Entry()
        self.sector_dmax_entry.set_text('infinity')
        tab.attach(self.sector_dmax_entry,1,2,1,2)
        
        l=gtk.Label('Sector width (deg):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        self.sector_width_entry=gtk.Entry()
        self.sector_width_entry.set_text('20')
        tab.attach(self.sector_width_entry,1,2,2,3)
        
        b=gtk.Button('Shade range on plot')
        tab.attach(b,0,2,3,4)
        b.connect('clicked',self.shade_range,'sector')
        
        ### Azimuthal beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Azimuthal'))
        l=gtk.Label('Min. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.azim_dmin_entry=gtk.Entry()
        self.azim_dmin_entry.set_text('0')
        tab.attach(self.azim_dmin_entry,1,2,0,1)

        l=gtk.Label('Max. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.azim_dmax_entry=gtk.Entry()
        self.azim_dmax_entry.set_text('infinity')
        tab.attach(self.azim_dmax_entry,1,2,1,2)
        
        l=gtk.Label('Number of bins:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        self.azim_Ntheta_entry=gtk.Entry()
        self.azim_Ntheta_entry.set_text('50')
        tab.attach(self.azim_Ntheta_entry,1,2,2,3)
        b=gtk.Button('Shade range on plot')
        tab.attach(b,0,2,3,4)
        b.connect('clicked',self.shade_range,'azim')
        
        ### Radial peak beam finding
        tab=gtk.Table()
        self.notebook.append_page(tab,gtk.Label('Radial peak'))
        l=gtk.Label('Min. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.rad_dmin_entry=gtk.Entry()
        self.rad_dmin_entry.set_text('0')
        tab.attach(self.rad_dmin_entry,1,2,0,1)

        l=gtk.Label('Max. radius (pixel):')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        self.rad_dmax_entry=gtk.Entry()
        self.rad_dmax_entry.set_text('infinity')
        tab.attach(self.rad_dmax_entry,1,2,1,2)
        
        l=gtk.Label('Optimize:')
        l.set_alignment(0,0.5)
        tab.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        try:
            self.rad_opttype=gtk.ComboBoxText()
        except AttributeError:
            self.rad_opttype=gtk.combo_box_new_text()
        self.rad_opttype.append_text('amplitude')
        self.rad_opttype.append_text('hwhm')
        self.rad_opttype.set_active(0)
        tab.attach(self.rad_opttype,1,2,2,3)
        
        b=gtk.Button('Shade range on plot')
        tab.attach(b,0,2,3,4)
        b.connect('clicked',self.shade_range,'radial')
        
        hb=gtk.HBox()
        self.pack_start(hb,False,True)
        l=gtk.Label('Beam pos.:')
        l.set_alignment(0,0.5)
        hb.pack_start(l,False,True)
        self.beamposx_spin=gtk.SpinButton()
        hb.pack_start(self.beamposx_spin,True,True)
        self.beamposy_spin=gtk.SpinButton()
        hb.pack_start(self.beamposy_spin,True,True)
        for sb in [self.beamposx_spin, self.beamposy_spin]:
            sb.set_range(-1e6,1e6)
            sb.set_digits(3)
            sb.set_increments(0.1,1)
            sb.set_numeric(True)
            sb.connect('value-changed',self.beampos_changed)
        self.auto_apply_cb=gtk.CheckButton('Auto-apply beam pos.')
        self.auto_apply_cb.set_active(True)
        self.pack_start(self.auto_apply_cb,False,True)
        hbb=gtk.HButtonBox()
        self.pack_end(hbb)
        b=gtk.Button(stock=gtk.STOCK_EXECUTE)
        b.connect('clicked',self.findcenter)
        hbb.add(b)
        b=gtk.Button(stock=gtk.STOCK_HELP)
        b.connect('clicked',self.helpmessage)
        hbb.add(b)
        b=gtk.Button(stock=gtk.STOCK_APPLY)
        b.connect('clicked',self.update_dataset)
        hbb.add(b)
        b=gtk.Button(label='QC')
        b.connect('clicked',self.testbeampos)
        hbb.add(b)
    def beampos_changed(self,spinbutton=None):
        pass
    def centeringmode(self):
        return self.notebook.get_tab_label_text(self.notebook.get_nth_page(self.notebook.get_current_page()))
    def _findbeam_click_handler(self,event):
        if event.button==1:
            self._click_pos=[event.ydata, event.xdata]
            self.matrix_source.canvas.mpl_disconnect(self._click_cid)
            del self._click_cid
        return True
    def findcenter(self,widget):
        d=gtk.Dialog('Centering...',self.get_toplevel(),gtk.DIALOG_DESTROY_WITH_PARENT|gtk.DIALOG_MODAL,(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL))
        pb=gtk.ProgressBar()
        d.vbox.pack_start(pb)
        def breakloop(*args,**kwargs):
            self._break=True
        self._break=False
        d.action_area.get_children()[0].connect('clicked',breakloop)
        pb.set_text('Finding beam position...')
        def callback(selfobject=self):
            if selfobject._break:
                raise sastool_break_loop('User break')
            pb.pulse()
            while gtk.events_pending():
                gtk.main_iteration()
        data=self.matrix_source.data
        update=self.auto_apply_cb.get_active()
        try:
            if data is None:
                raise ValueError('No data loaded yet!')
            if self.centeringmode()=='Semitransp.':
                bs_area=[float(x.get_text()) for x in [self.pri_rmin_entry,
                                                       self.pri_rmax_entry,
                                                       self.pri_cmin_entry,
                                                       self.pri_cmax_entry]]
                bcx,bcy=data.find_beam_semitransparent(bs_area, update)
            elif self.centeringmode()=='Click':
                try:
                    self.matrix_source.enable_toolbar(False)
                    prevtitle=self.get_toplevel().get_title()
                    self._click_cid=self.matrix_source.canvas.mpl_connect('button_press_event',self._findbeam_click_handler)
                    if hasattr(self,'_click_pos'):
                        del self._click_pos
                    while True:
                        self.get_toplevel().set_title('Click on the beam position in the scattering pattern!')
                        self.matrix_source.fig.waitforbuttonpress(1)
                        if hasattr(self,'_click_pos'):
                            break
                        self.get_toplevel().set_title(prevtitle)
                        self.matrix_source.fig.waitforbuttonpress(1)
                        if hasattr(self,'_click_pos'):
                            break
                    bcx,bcy=self._click_pos
                    if update:
                        self.matrix_source.get_data().update_beampos((bcx,bcy),'point-and-click')
                    del self._click_pos
                except:
                    raise
                finally:
                    self.matrix_source.enable_toolbar(True)
                    self.get_toplevel().set_title(prevtitle)
            elif self.centeringmode()=='Gravity':
                bcx,bcy=data.find_beam_gravity()
            elif self.centeringmode()=='Sectors':
                dmin=float(self.sector_dmin_entry.get_text())
                dmax=float(self.sector_dmax_entry.get_text())
                sector_width=float(self.sector_width_entry.get_text())*np.pi/180.
                d.show_all()
                bcx,bcy=data.find_beam_slices(dmin,dmax,sector_width, update, callback)
            elif self.centeringmode()=='Azimuthal':
                dmin=float(self.azim_dmin_entry.get_text())
                dmax=float(self.azim_dmax_entry.get_text())
                Ntheta=float(self.azim_Ntheta_entry.get_text())
                d.show_all()
                bcx,bcy=data.find_beam_azimuthal_fold(Ntheta,dmin,dmax,update, callback)
            elif self.centeringmode()=='Radial peak':
                dmin=float(self.rad_dmin_entry.get_text())
                dmax=float(self.rad_dmax_entry.get_text())
                opt=self.rad_opttype.get_active_text()
                d.show_all()
                bcx,bcy=data.find_beam_radialpeak(dmin,dmax,opt,update=update,callback=callback)
        except sastool_break_loop:
            #User break in iteration
            d.hide()
            md=gtk.MessageDialog(self.get_toplevel(), 
                                 gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,
                                 gtk.MESSAGE_INFO, gtk.BUTTONS_OK, 'User break in centering algorithm.')
            md.set_title('User break')
            md.run()
            md.destroy()
        except classes.SASMaskException as ex:
            md=gtk.MessageDialog(self.get_toplevel(), 
                                 gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,
                                 gtk.MESSAGE_ERROR, gtk.BUTTONS_OK, ex.message)
            md.set_title('Masking error')
            md.run()
            md.destroy()
        except Exception as ex:
            #other error
            md=gtk.MessageDialog(self.get_toplevel(), 
                                 gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,
                                 gtk.MESSAGE_ERROR, gtk.BUTTONS_OK, ex.message)
            md.set_title(str(type(ex)))
            md.run()
            md.destroy()
        else:
            #Success, should return the beam center somewhere
            self.beamposx_spin.set_value(bcx)
            self.beamposy_spin.set_value(bcy)
        finally:
            self.matrix_source.refresh_stats()
            d.destroy()
        return True
    def helpmessage(self,widget):
        if self.centeringmode()=='Semitransp.':
            title="Semi-transparent algorithm"
            msg="If the scattering pattern was recorded using a semi-transparent \
beamstop, you can zoom to the beam area and a weighting algorithm \
can find the beam position for you."""
            goodfor='Images with visible attenuated beam'
        elif self.centeringmode()=='Click':
            title="By-hand centering"
            msg="After pressing 'Execute', click the image to select the beam \
position by hand."
            goodfor='Quick-and-dirty beam finding, for first approximation.'
        elif self.centeringmode()=='Gravity':
            title="Gravity method"
            msg="Finds the beam center by finding the center of gravity in \
both directions."
            goodfor='Images where not much masking is needed. Marked Bragg-rings improve precision.'
        elif self.centeringmode()=='Sectors':
            title="Four-sector algorithm"
            msg="Minimize the difference between opposite sector pairs \
(diagonal) on the scattering image."
            goodfor='High accuracy isotropic images where the full azimuth range is available.'
        elif self.centeringmode()=='Azimuthal':
            title="Azimuthal algorithm"
            msg="Find the beam position in such a way that the azimuthal curve \
has pi periodicity (I(chi) overlaps with I(chi+pi))"
            goodfor="Images having a point of inversion and fully available azimuth range."
        elif self.centeringmode()=='Radial peak':
            title="Radial peak algorithm"
            msg="Minimize the FWHM or maximize the amplitude of a peak in the \
radial intensity curve."
            goodfor="Images possessing at least one well-pronounced Bragg-peak (small anisotropy should be OK)."
        else:
            title=self.centeringmode()
            msg="""This centering algorithm has not yet been documented.
            """
            goodfor="None"
        md=gtk.MessageDialog(parent=self.get_toplevel(), 
                             flags=gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT,
                             type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK)
        md.set_markup('<b>Description:</b>\n%s\n<b>Good for:</b>\n%s\n'%(msg,goodfor))
        md.set_title(title)
        md.run()
        md.destroy()
        return True
    def update_dataset(self,widget):
        bcx=self.beamposx_spin.get_value()
        bcy=self.beamposy_spin.get_value()
        self.matrix_source.data.update_beampos((bcx,bcy))
        self.matrix_source.refresh_stats()
    def get_pri_from_zoom(self,widget):
        colmin,colmax,rowmin,rowmax=self.matrix_source.get_zoom()
        self.pri_cmin_entry.set_text('%.4f'%min(colmin,colmax))
        self.pri_cmax_entry.set_text('%.4f'%max(colmin,colmax))
        self.pri_rmin_entry.set_text('%.4f'%min(rowmin,rowmax))
        self.pri_rmax_entry.set_text('%.4f'%max(rowmin,rowmax))
        return True
    def shade_range(self,widget,what):
        if not self.matrix_source.fig.axes:
            return
        ax=self.matrix_source.fig.axes[0].axis()
        if what=='radial':
            dmin=float(self.rad_dmin_entry.get_text())
            dmax=float(self.rad_dmax_entry.get_text())
        elif what=='azim':
            dmin=float(self.azim_dmin_entry.get_text())
            dmax=float(self.azim_dmax_entry.get_text())
        elif what=='sector':
            dmin=float(self.sector_dmin_entry.get_text())
            dmax=float(self.sector_dmax_entry.get_text())
        else:
            raise ValueError(what)
        for d in set(np.arange(dmin,dmax,10)).union({dmin,dmax}):
            t=np.linspace(0,2*np.pi,d*np.pi*2)
            x=self.matrix_source.get_data().header['BeamPosY']+d*np.cos(t)
            y=self.matrix_source.get_data().header['BeamPosX']+d*np.sin(t)
            self.matrix_source.fig.axes[0].plot(x,y,'w-')
            del t,x,y
        self.matrix_source.fig.axes[0].axis(ax)
        self.matrix_source.canvas.draw()
    def testbeampos(self,widget):
        """Plot a few beam position assessment images"""
        data=self.matrix_source.get_data()
        mat=self.matrix_source.get_plotted_matrix()
        mask=data.mask.mask
        bcx=data.header['BeamPosX']
        bcy=data.header['BeamPosY']
        maxpix=max(abs(mat.shape[0]-bcx),abs(mat.shape[1]-bcy),abs(bcx),abs(bcy))
        self.matrix_source.fig.clf()
        sector_width=float(self.sector_width_entry.get_text())*np.pi/180.
        # Normal image with cross-hair
        sp=self.matrix_source.fig.add_subplot(2,2,1)
        sp.imshow(mat,cmap=self.matrix_source.get_colormap(),interpolation='nearest')
        ax=sp.axis()
        sp.plot([0,mat.shape[1]],[bcx,bcx],'w-')
        sp.plot([bcy,bcy],[0,mat.shape[0]],'w-')
        sp.axis(ax)
        sp.set_title('Cross-hair')
        # Azimuthally regrouped
        sp=self.matrix_source.fig.add_subplot(2,2,2)
        r=np.arange(0,maxpix)
        phi=np.linspace(0,360,maxpix)
        azimmat=utils2d.integrate.polartransform(mat,r,phi,bcx,bcy)
        sp.imshow(azimmat,cmap=self.matrix_source.get_colormap(),interpolation='nearest')
        del azimmat
        del r
        del phi
        sp.set_title('Azimuthally regrouped')
        # Azimuthal scattering curve
        sp=self.matrix_source.fig.add_subplot(2,2,4)
        sp.set_title('Azimuthal scattering curves')
        ds=data.azimuthal_average(float(self.azim_dmin_entry.get_text()),
                                  float(self.azim_dmax_entry.get_text()),
                                  float(self.azim_Ntheta_entry.get_text()),
                                  pixel=True,
                                  matrix=self.matrix_source.get_data().get_matrix_name(),
                                  errormatrix=None)
        l1=ds.semilogy(sp,'b.-',label='<-')
        sp1=sp.twinx()
        l2=sp1.plot(ds.x,ds.Area,'r-',label='->')
        sp.set_xlabel('Azimuth angle')
        sp.set_ylabel('Intensity')
        sp1.set_ylabel('Effective area of bins (pixel)')
        sp.legend((l1,l2),('<-','->'),loc='best')
        # overlapping of slices
        sp=self.matrix_source.fig.add_subplot(2,2,3)
        sp.set_title('Overlap of diagonal sectors')
        ds1=data.sector_average(1*np.pi/4,sector_width,symmetric_sector=False,pixel=True,
                                matrix=self.matrix_source.get_data().get_matrix_name(),errormatrix=None)
        ds2=data.sector_average(3*np.pi/4,sector_width,symmetric_sector=False,pixel=True,
                                matrix=self.matrix_source.get_data().get_matrix_name(),errormatrix=None)
        ds3=data.sector_average(5*np.pi/4,sector_width,symmetric_sector=False,pixel=True,
                                matrix=self.matrix_source.get_data().get_matrix_name(),errormatrix=None)
        ds4=data.sector_average(7*np.pi/4,sector_width,symmetric_sector=False,pixel=True,
                                matrix=self.matrix_source.get_data().get_matrix_name(),errormatrix=None)
        ds1.semilogy(sp,'bo-',label='45$^\circ$')
        ds3.semilogy(sp,'b.-',label='-45$^\circ$')
        ds2.semilogy(sp,'ro-',label='135$^\circ$')
        ds4.semilogy(sp,'r.-',label='-135$^\circ$')
        sp.set_xlabel('pixels')
        sp.set_ylabel('Intensity')
        sp.legend(loc='best')
        del ds1, ds2, ds3, ds4
        self.matrix_source.canvas.draw()

class SAS2DGUI(gtk.Window):
    def __init__(self,*args,**kwargs):
        self.data=None
        gtk.Window.__init__(self,*args,**kwargs)
        self.set_title('SAS 2D GUI tool')
        hpaned=gtk.HPaned()
        self.add(hpaned)
        self.lefttoolbar=gtk.VPaned()
        hpaned.add1(self.lefttoolbar)
        vb=gtk.VBox()
        self.lefttoolbar.pack1(vb,True,False)
        
        # add a matplotlib figure
        figvbox = gtk.VBox()
        hpaned.add2(figvbox)
        self.fig = Figure(figsize = (0.5, 0.4), dpi = 72)
        self.axes=self.fig.add_subplot(111)

        self.canvas = FigureCanvasGTKAgg(self.fig)
        self.canvas.set_size_request(500, 200)
        figvbox.pack_start(self.canvas, True, True, 0)
        #self.canvas.mpl_connect('button_press_event', self._on_matplotlib_mouseclick)

        hb1 = gtk.HBox() # the toolbar below the figure
        figvbox.pack_start(hb1, False, True, 0)
        self.graphtoolbar = NavigationToolbar2GTKAgg(self.canvas, hpaned)
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
        self.masker=SAS2DMasker(matrix_source=self)
        ex.add(self.masker)
        
        ex=gtk.Expander(label='Centering')
        vb.pack_start(ex,False,True)
        self.centerer=SAS2DCenterer(matrix_source=self)
        ex.add(self.centerer)
        
        
        integrator=gtk.Expander(label='Radial averaging')
        vb.pack_start(integrator,False,True)
        
        self.statistics=SAS2DStatistics()
        self.lefttoolbar.pack2(self.statistics,True,False)
        
        self.show_all()
        self.hide()
    def enable_toolbar(self,state):
        self.lefttoolbar.set_sensitive(state)
    def get_zoom(self):
        if not self.fig.axes:
            self.fig.add_subplot(111)
            self.canvas.draw()
        return self.fig.axes[0].axis()
    def get_data(self):
        return self.data
    def get_matrix(self):
        if self.data is not None:
            matrixtype=[x for x in self.data.matrices.keys() if self.data.matrices[x]==self.plotter.matrixtype.get_active_text()][0]
            return self.data.__getattribute__(matrixtype)
        else:
            return None
    def get_plotted_matrix(self):
        if hasattr(self.plotter,'mat'):
            return self.plotter.mat
        else:
            return self.get_matrix()
    def get_colormap(self):
        if hasattr(self.plotter,'cmap'):
            return self.plotter.cmap
        else:
            return matplotlib.cm.jet
    def replot(self):
        if self.data is not None:
            self.plotter.plot2d(self.data)
    def refresh_stats(self,widget=None):
        if self.data is None:
            return
        mat=self.data.get_matrix(self.plotter.get_matrixtype())
        sumdata=np.nansum(mat)
        Nandata=(-np.isfinite(mat)).sum()
        maxdata=np.nanmax(mat)
        mindata=np.nanmin(mat)
        if self.data.mask is not None:
            maxmasked=np.nanmax(mat*self.data.mask.mask)
            minmasked=np.nanmin(mat*self.data.mask.mask)
            summasked=np.nansum(mat*self.data.mask.mask)
        else:
            maxmasked='N/A'
            minmasked='N/A'
            summasked='N/A'
        shape=mat.shape
        self.statistics.update_table(collections.OrderedDict([('FSN',str(self.data.header['FSN'])),
                                                              ('Title',self.data.header['Title']),
                                                              ('Beam row, col',str((self.data.header['BeamPosX'],self.data.header['BeamPosY']))),
                                                              ('Shape','%d x %d'%shape),
                                                              ('Total counts',sumdata),
                                                              ('Invalid pixels',Nandata),
                                                              ('Max. count',maxdata),
                                                              ('Min. count',mindata),
                                                              ('Total counts (mask)',summasked),
                                                              ('Max. count (mask)',maxmasked),
                                                              ('Min. count (mask)',minmasked)]))
        
    def file_opened(self,widget,exposition):
        if self.data is not None:
            del self.data
        self.data=exposition
        self.plotter.update_matrixtype(self.data)
        if self.data.mask is None:
            self.masker.updatemaskindata()
        self.refresh_stats()
        if self.data.mask is not None:
            self.masker.setmask(self.data.mask)
        self.replot()
    def __del__(self):
        del self.data
        super(SAS2DGUI,self).__del__()
        
def SAS2DGUI_run():
    w = SAS2DGUI()
    def f(widget,event,*args,**kwargs):
        widget.destroy()
        del widget
    w.connect('delete-event',f)
    w.show()
    
    