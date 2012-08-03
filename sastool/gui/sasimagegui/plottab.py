import gtk

from PyGTKCallback import PyGTKCallback
@PyGTKCallback
class PlotTab(gtk.HBox):
    def __init__(self):
        gtk.HBox.__init__(self)
        tb = gtk.Toolbar()
        tb.set_show_arrow(False)
        tb.set_style(gtk.TOOLBAR_BOTH)
        self.pack_start(tb, False, True)

        b = gtk.ToolButton(gtk.STOCK_CLEAR)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'clear-graph')

        b = gtk.ToolButton(gtk.STOCK_REFRESH)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'refresh-graph')

    def on_button_clicked(self, button, arg):
        self.emit(arg)
