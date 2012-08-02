import gtk
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg, NavigationToolbar2GTKAgg

from filetab import FileTab

class SASImageGuiMain(gtk.Window):
    _Ninstances = 0
    def __init__(self):
        SASImageGuiMain._Ninstances += 1
        gtk.Window.__init__(self, gtk.WINDOW_TOPLEVEL)

        self.connect('delete-event', self.on_delete)

        vbox = gtk.VBox()
        self.add(vbox)
        self.ribbon = gtk.Notebook()
        vbox.pack_start(self.ribbon, False)
        self.ribbon.set_scrollable(True)
        self.ribbon.popup_enable()

        self.ribbon_File = FileTab()
        self.ribbon.append_page(self.ribbon_File, gtk.Label('File'))
        self.ribbon.set_tab_detachable(self.ribbon_File, True)
        self.ribbon_File.connect('new-clicked', self.on_file_new)

        self.ribbon_Plot = gtk.VBox()
        self.ribbon.append_page(self.ribbon_Plot, gtk.Label('Plot'))

        self.ribbon_Mask = gtk.VBox()
        self.ribbon.append_page(self.ribbon_Mask, gtk.Label('Mask'))

        self.ribbon_Centering = gtk.VBox()
        self.ribbon.append_page(self.ribbon_Centering, gtk.Label('Centering'))

        self.ribbon_Integrate = gtk.VBox()
        self.ribbon.append_page(self.ribbon_Integrate, gtk.Label('Integrate'))

        for k in self.ribbon.get_children():
            self.ribbon.set_tab_detachable(k, True)
            self.ribbon.set_tab_reorderable(k, True)

        self.fig = Figure(figsize=(0.5, 0.4), dpi=72)
        self.axes = self.fig.add_subplot(111)

        self.canvas = FigureCanvasGTKAgg(self.fig)
        self.canvas.set_size_request(800, 300)
        vbox.pack_start(self.canvas, True, True, 0)

        self.graphtoolbar = NavigationToolbar2GTKAgg(self.canvas, vbox)
        vbox.pack_start(self.graphtoolbar, False)

        self.statusbar = gtk.Statusbar()
        vbox.pack_start(self.statusbar, False)

        self.show_all()
        self.hide()
    def on_file_new(self, widget):
        newinstance = SASImageGuiMain()
        newinstance.show_all()
    def on_delete(self, widget, event, *args):
        SASImageGuiMain._Ninstances -= 1
        if SASImageGuiMain._Ninstances == 0:
            gtk.main_quit()
        return False
