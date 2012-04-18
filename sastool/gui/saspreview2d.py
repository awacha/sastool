'''
Created on Feb 14, 2012

@author: andris
'''


import gtk
import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.backends
if matplotlib.backends.backend.upper() != 'GTKAGG':
    raise ImportError('Sastool.gui works only with the GTK backend of \
Matplotlib')

from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg, \
NavigationToolbar2GTKAgg
from matplotlib.figure import Figure

class SAS2DLoader(gtk.Notebook):
    def __init__(self,*args,**kwargs):
        gtk.Notebook.__init__(self)
        b1org=gtk.Table()
        self.append_page(b1org,gtk.Label('B1 org'))
        
        l=gtk.Label('Filename format')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,1,0,1,gtk.FILL,gtk.FILL)
        self.fileformat_entry=gtk.Entry()
        self.fileformat_entry.set_text('org_%05d')
        b1org.attach(self.fileformat_entry,1,2,0,1,gtk.EXPAND|gtk.FILL,gtk.EXPAND|gtk.FILL)
        
        l=gtk.Label('Image format')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,1,1,2,gtk.FILL,gtk.FILL)
        try:
            self.imageformat=gtk.ComboBoxText()
        except AttributeError:
            self.imageformat=gtk.combo_box_new_text()
        self.imageformat.append_text('CBF')
        self.imageformat.append_text('TIF(F)')
        self.imageformat.append_text('DAT(.gz)')
        self.imageformat.set_active(0)
        b1org.attach(self.imageformat,1,2,1,2)
        
        l=gtk.Label('FSN')
        l.set_alignment(0,0.5)
        b1org.attach(l,0,1,2,3,gtk.FILL,gtk.FILL)
        self.fsn_spinbutton=gtk.SpinButton(gtk.Adjustment(0,0,1e10,1,10),digits=0)
        b1org.attach(self.fsn_spinbutton,1,2,2,3)
        
        b=gtk.Button(stock=gtk.STOCK_OPEN)
        b1org.attach(b,0,2,3,4,gtk.EXPAND|gtk.FILL,gtk.EXPAND|gtk.FILL)
        
        
        vb=gtk.VBox()
        self.append_page(vb,gtk.Label('B1 processed'))
        
        vb=gtk.VBox()
        self.append_page(vb,gtk.Label('ID02'))
        
class SAS2DGUI(gtk.Window):
    def __init__(self,*args,**kwargs):
        gtk.Window.__init__(self,*args,**kwargs)
        hb=gtk.HBox()
        self.add(hb)
        #build the toolbar on the left pane
        vb=gtk.VBox()
        hb.pack_start(vb,False,True)
        ex=gtk.Expander(label='Load measurement')
        vb.pack_start(ex)
        self.loader=SAS2DLoader()
        ex.add(self.loader)
        
        masker=gtk.Expander(label='Masking')
        vb.pack_start(masker)
        integrator=gtk.Expander(label='Radial averaging')
        vb.pack_start(integrator)
        centerer=gtk.Expander(label='Centering')
        vb.pack_start(centerer)
        
        # add a matplotlib figure
        figvbox = gtk.VBox()
        hb.pack_start(figvbox)
        self.fig = Figure(figsize = (0.5, 0.4), dpi = 72)
        self.fig.add_subplot(111)

        self.canvas = FigureCanvasGTKAgg(self.fig)
        self.canvas.set_size_request(300, 200)
        figvbox.pack_start(self.canvas, True, True, 0)
        #self.canvas.mpl_connect('button_press_event', self._on_matplotlib_mouseclick)

        hb1 = gtk.HBox() # the toolbar below the figure
        figvbox.pack_start(hb1, False, True, 0)
        self.graphtoolbar = NavigationToolbar2GTKAgg(self.canvas, hb)
        hb1.pack_start(self.graphtoolbar, True, True, 0)
        
        self.show_all()
        self.hide()


def SAS2DGUI_run():
    w = SAS2DGUI()
    w.show()
    