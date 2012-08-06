import gtk

from PyGTKCallback import PyGTKCallback
@PyGTKCallback
class MaskTab(gtk.HBox):
    def __init__(self):
        gtk.HBox.__init__(self)
        tb = gtk.Toolbar()
        tb.set_show_arrow(False)
        tb.set_style(gtk.TOOLBAR_BOTH)
        self.pack_start(tb, False, True)

        b = gtk.ToolButton(gtk.STOCK_NEW)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'new-mask')

        b = gtk.ToolButton(gtk.STOCK_EDIT)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'edit-mask')

    def on_button_clicked(self, widget, argument):
        self.emit(argument)
