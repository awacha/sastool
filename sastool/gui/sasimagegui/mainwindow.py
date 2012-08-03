import gtk
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg, NavigationToolbar2GTKAgg
from matplotlib import pyplot as plt

from filetab import FileTab
from plottab import PlotTab
from masktab import MaskTab
from centeringtab import CenteringTab
from integratetab import IntegrateTab

class SASImageGuiMain(gtk.Window):
    _instances = []
    def __init__(self):
        SASImageGuiMain._instances.append(self)
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
        self.ribbon_File.connect('new-clicked', self.on_file, 'newwindow')
        self.ribbon_File.connect('close-clicked', self.on_file, 'closewindow')
        self.ribbon_File.connect('quit-clicked', self.on_file, 'quitprogram')
        self.ribbon_File.connect('opened', self.on_file_opened)

        self.ribbon_Plot = PlotTab()
        self.ribbon.append_page(self.ribbon_Plot, gtk.Label('Plot'))
        self.ribbon_Plot.connect('clear-graph', self.on_plot, 'clear-graph')

        self.ribbon_Mask = MaskTab()
        self.ribbon.append_page(self.ribbon_Mask, gtk.Label('Mask'))

        self.ribbon_Centering = CenteringTab()
        self.ribbon.append_page(self.ribbon_Centering, gtk.Label('Centering'))

        self.ribbon_Integrate = IntegrateTab()
        self.ribbon.append_page(self.ribbon_Integrate, gtk.Label('Integrate'))

        for k in self.ribbon.get_children():
            self.ribbon.set_tab_detachable(k, True)
            self.ribbon.set_tab_reorderable(k, True)
            k.connect('error', self.on_error)
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
    def on_file(self, widget, argument):
        if argument == 'newwindow':
            newinstance = SASImageGuiMain()
            newinstance.show_all()
        elif argument == 'closewindow':
            self.destroy()
            self.on_delete(None, None)
        elif argument == 'quitprogram':
            gtk.main_quit()
        return True
    def on_delete(self, widget, event, *args):
        SASImageGuiMain._instances.remove(self)
        if not SASImageGuiMain._instances:
            gtk.main_quit()
        return False
    def on_file_opened(self, widget, data):
        self.statusbar.remove_all(self.statusbar.get_context_id('sastool'))
        self.statusbar.push(self.statusbar.get_context_id('sastool'), 'File opened.')
        self.on_plot(None, 'refresh-graph')
        return True
    def on_error(self, widget, error):
        self.statusbar.remove_all(self.statusbar.get_context_id('sastool'))
        self.statusbar.push(self.statusbar.get_context_id('sastool'), 'Error: ' + unicode(error))
        return True
    def on_plot(self, widget, argument):
        if argument == 'clear-graph':
            self.fig.clf()
            self.axes = self.fig.add_subplot(1, 1, 1)
        if argument == 'refresh-graph':
            self.axes.cla()
            self.data.plot2d(axes=self.axes)
        self.canvas.draw_idle()
        return True
    def get_data(self):
        return self.ribbon_File.data
    data = property(get_data)
