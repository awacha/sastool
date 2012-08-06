import gtk
import os
import gobject

from PyGTKCallback import PyGTKCallback
from ... import misc
from .. import patheditor
from ... import classes

@PyGTKCallback
class FileTab(gtk.HBox):
    def __init__(self, searchpath=None):
        gtk.HBox.__init__(self)

        self.oldexptype = None
        if searchpath is None:
            searchpath = misc.sastoolrc.get('misc.searchpath')
        self.searchpath = misc.searchpath.SearchPath(searchpath)
        self.data = None

        tb = gtk.Toolbar()
        tb.set_show_arrow(False)
        tb.set_style(gtk.TOOLBAR_BOTH)
        self.pack_start(tb, False, True)
        b = gtk.ToolButton(gtk.STOCK_NEW)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'new-clicked')
        b = gtk.ToolButton(gtk.STOCK_CLOSE)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'close-clicked')
        b = gtk.ToolButton(gtk.STOCK_QUIT)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'quit-clicked')
        b = gtk.ToolButton(None, "Adjust path")
        b.set_stock_id(gtk.STOCK_DIRECTORY)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'adjust-path-clicked')
        frame = gtk.Frame()
        self.pack_start(frame, False)
        tab = gtk.Table()
        frame.add(tab)

        tablecolumn = 0
        tablerow = 0
        l = gtk.Label('FSN:');  l.set_alignment(0, 0.5);  tab.attach(l, 2 * tablecolumn + 0, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.fsn_entry = gtk.SpinButton()
        tab.attach(self.fsn_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.fsn_entry.set_range(0, 10000000000)
        self.fsn_entry.set_digits(0)
        self.fsn_entry.set_value(0)
        self.fsn_entry.set_increments(1, 10)

        tablerow += 1
        l = gtk.Label('Experiment type:');  l.set_alignment(0, 0.5);  tab.attach(l, 2 * tablecolumn + 0, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.exptype_entry = gtk.combo_box_new_text()
        tab.attach(self.exptype_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.exptype_entry.connect('changed', self.on_exptype_changed)
        self.exptype_entry.append_text('ESRF ID02')
        self.exptype_entry.append_text('B1 org')
        self.exptype_entry.append_text('B1 int2dnorm')
        self.exptype_entry.append_text('PAXE')
        self.exptype_entry.append_text('HDF5')

        tablerow += 1
        l = gtk.Label('Filename format:');  l.set_alignment(0, 0.5);  tab.attach(l, 2 * tablecolumn + 0, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.filenameformat_entry = gtk.Entry()
        tab.attach(self.filenameformat_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)


        tablecolumn += 1
        tablerow = 0
        l = gtk.Label('Mask file name:');  l.set_alignment(0, 0.5);  tab.attach(l, 2 * tablecolumn + 0, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.maskname_entry = gtk.Entry()
        tab.attach(self.maskname_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        self.loadmask_checkbox = gtk.CheckButton('Load mask')
        tab.attach(self.loadmask_checkbox, 2 * tablecolumn + 0, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        l = gtk.Label('Headername format:');  l.set_alignment(0, 0.5);  tab.attach(l, 2 * tablecolumn + 0, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.headernameformat_entry = gtk.Entry()
        tab.attach(self.headernameformat_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablecolumn += 1
        b = gtk.Button(stock=gtk.STOCK_OPEN)
        b.set_image_position(gtk.POS_TOP)
        tab.attach(b, 2 * tablecolumn + 0, 2 * tablecolumn + 2, 0, 3)
        b.connect('clicked', self.on_open)

        self.exptype_entry.set_active(0)


        self.show_all()
    def on_button_clicked(self, button, argument):
        if argument == 'adjust-path-clicked':
            pe = patheditor.PathEditor(self, self.searchpath)
            try:
                ret = pe.run()
                if ret == gtk.RESPONSE_OK:
                    pe.update_search_path()
            finally:
                pe.destroy()
            self.emit(argument, self.searchpath)
        else:
            self.emit(argument)
    def on_exptype_changed(self, cbox):
        exptype = cbox.get_active_text().replace(' ', '_')
        if self.oldexptype is not None:
            misc.sastoolrc.set('gui.sasimagegui.file.headerformat_%s' % self.oldexptype, self.headernameformat_entry.get_text())
            misc.sastoolrc.set('gui.sasimagegui.file.fileformat_%s' % self.oldexptype, self.filenameformat_entry.get_text())
        try:
            self.headernameformat_entry.set_text(misc.sastoolrc.get('gui.sasimagegui.file.headerformat_%s' % exptype))
        except KeyError:
            self.headernameformat_entry.set_text('')
        try:
            self.filenameformat_entry.set_text(misc.sastoolrc.get('gui.sasimagegui.file.fileformat_%s' % exptype))
        except KeyError:
            self.filenameformat_entry.set_text('')
        self.oldexptype = exptype

    def on_open(self, button):
        maskname = self.maskname_entry.get_text()
        if not maskname:
            maskname = None
        try:
            self.data = classes.SASExposure(self.filenameformat_entry.get_text(),
                                            self.fsn_entry.get_value_as_int(),
                                            dirs=self.searchpath,
                                            maskfile=maskname,
                                            load_mask=self.loadmask_checkbox.get_active(),
                                            experiment_type=self.exptype_entry.get_active_text().replace(' ', '_'),
                                            fileformat=os.path.splitext(os.path.split(self.filenameformat_entry.get_text())[-1])[0],
                                            logfileformat=os.path.splitext(self.headernameformat_entry.get_text())[0],
                                            logfileextn=os.path.splitext(self.headernameformat_entry.get_text())[1])
        except IOError as ioe:
            self.emit('error', ioe)
        else:
            gobject.idle_add(self.call_callbacks_on_open)
        return True
    def call_callbacks_on_open(self):
        if self.data:
            self.emit('opened', self.data)
        return False
