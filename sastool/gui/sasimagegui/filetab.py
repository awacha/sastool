import gtk

from PyGTKCallback import PyGTKCallback

@PyGTKCallback
class FileTab(gtk.HBox):
    def __init__(self):
        gtk.HBox.__init__(self)
        b = gtk.Button(stock=gtk.STOCK_NEW)
        self.pack_start(b)
        b.connect('clicked', self.on_new_clicked)
        self.show_all()
    def on_new_clicked(self, button):
        self.emit('new-clicked')
