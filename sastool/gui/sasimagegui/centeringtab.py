import gtk
import pkg_resources

xhair = gtk.IconSet(gtk.gdk.pixbuf_new_from_file(pkg_resources.resource_filename('sastool', 'resource/icons/crosshair.png')))
gtk.stock_add([('sastool_crosshair', 'Set from graph', 0, 0, 'EN_US')])
ifa = gtk.IconFactory()
ifa.add('sastool_crosshair', xhair)
ifa.add_default()

from PyGTKCallback import PyGTKCallback
@PyGTKCallback
class CenteringTab(gtk.HBox):
    def __init__(self):
        gtk.HBox.__init__(self)
        tb = gtk.Toolbar()
        tb.set_show_arrow(False)
        tb.set_style(gtk.TOOLBAR_BOTH)
        self.pack_start(tb, False, True)

        b = gtk.ToolButton('sastool_crosshair')
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'crosshair')

        b = gtk.ToolButton(gtk.STOCK_EXECUTE)
        tb.insert(b, -1)
        b.connect('clicked', self.on_button_clicked, 'execute')

        frame = gtk.Frame()
        self.pack_start(frame, False)
        tab = gtk.Table()
        frame.add(tab)

        tablerow = 0
        tablecolumn = 0

        l = gtk.Label('Method:'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.method_combo = gtk.combo_box_new_text()
        tab.attach(self.method_combo, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.method_combo.append_text('entered manually')
        self.method_combo.append_text('azimuthal')
        self.method_combo.append_text('gravity')
        self.method_combo.append_text('radial peak height')
        self.method_combo.append_text('radial peak width')
        self.method_combo.append_text('semitransparent')
        self.method_combo.append_text('slices')
        self.method_combo.connect('changed', self.on_method_changed)

        tablerow += 1
        l = gtk.Label('Min. pixel:'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.minpixel_entry = gtk.Entry()
        tab.attach(self.minpixel_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        l = gtk.Label('Max. pixel:'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.maxpixel_entry = gtk.Entry()
        tab.attach(self.maxpixel_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablecolumn += 1
        tablerow = 0

        l = gtk.Label(u'Nr of ' + unichr(0x03c6) + u' bins:'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.nphi_entry = gtk.Entry()
        tab.attach(self.nphi_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        l = gtk.Label(u'Pixel extent:'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.extent_entry = gtk.Entry()
        tab.attach(self.extent_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        l = gtk.Label(u'Sector width (deg):'); l.set_alignment(0, 0.5)
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.sectorwidth_entry = gtk.Entry()
        tab.attach(self.sectorwidth_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        self.method_combo.set_active(0)

        frame = gtk.Frame(label='Beam position:')
        self.pack_start(frame, False)
        tab = gtk.Table()
        frame.add(tab)

        l = gtk.Label('Row:')
        tab.attach(l, 0, 1, 0, 1, gtk.FILL, gtk.FILL)
        self.beamposx_entry = gtk.Entry()
        sr = self.beamposx_entry.get_size_request()
        self.beamposx_entry.set_size_request(100, sr[1])
        self.beamposx_entry.set_text('0')
        tab.attach(self.beamposx_entry, 1, 2, 0, 1)
        b = gtk.Button('Get')
        tab.attach(b, 2, 3, 0, 1, gtk.FILL, gtk.FILL)
        b.connect('clicked', self.pollbeamposx)


        l = gtk.Label('Column:')
        tab.attach(l, 0, 1, 1, 2, gtk.FILL, gtk.FILL)
        self.beamposy_entry = gtk.Entry()
        sr = self.beamposy_entry.get_size_request()
        self.beamposy_entry.set_size_request(100, sr[1])
        self.beamposy_entry.set_text('0')
        tab.attach(self.beamposy_entry, 1, 2, 1, 2)
        b = gtk.Button('Get')
        tab.attach(b, 2, 3, 1, 2, gtk.FILL, gtk.FILL)
        b.connect('clicked', self.pollbeamposy)
    def pollbeamposx(self, *args):
        print self.beamposx_entry.size_request()
        try:
            self.beamposx_entry.set_text(unicode(self.get_toplevel().data.header['BeamPosX']))
        except AttributeError:
            pass
    def pollbeamposy(self, *args):
        try:
            self.beamposy_entry.set_text(unicode(self.get_toplevel().data.header['BeamPosY']))
        except AttributeError:
            pass

    def on_button_clicked(self, widget, argument):
        if argument == 'crosshair':
            self.emit('crosshair', 'crosshair')
        elif argument == 'execute':
            method = self.method_combo.get_active_text()
            if method == 'entered manually':
                self.emit('manual-beampos', 'manual-beampos', float(self.beamposx_entry.get_text()),
                          float(self.beamposy_entry.get_text()))
            elif method == 'azimuthal':
                self.emit('docentering', 'azimuthal_fold', float(self.nphi_entry.get_text()),
                          float(self.minpixel_entry.get_text()),
                          float(self.maxpixel_entry.get_text()))
            elif method == 'gravity':
                self.emit('docentering', 'gravity')
            elif method == 'radial peak height':
                self.emit('docentering', 'radialpeak', float(self.minpixel_entry.get_text()),
                          float(self.maxpixel_entry.get_text()), 'amplitude',
                          float(self.extent_entry.get_text()))
            elif method == 'radial peak width':
                self.emit('docentering', 'radialpeak', float(self.minpixel_entry.get_text()),
                          float(self.maxpixel_entry.get_text()), 'hwhm',
                          float(self.extent_entry.get_text()))
            elif method == 'semitransparent':
                ax = self.get_toplevel().fig.axes[0].axis()
                self.emit('docentering', 'semitransparent', (ax[2], ax[3], ax[0], ax[1]))
            elif method == 'slices':
                self.emit('docentering', 'slices', float(self.minpixel_entry.get_text()),
                          float(self.maxpixel_entry.get_text()),
                          float(self.sectorwidth_entry.get_text()))
        return True

    def on_method_changed(self, combobox):
        if combobox.get_active_text() in ['azimuthal', 'radial peak height', 'radial peak width', 'slices']:
            self.minpixel_entry.set_sensitive(True)
            self.maxpixel_entry.set_sensitive(True)
            self.extent_entry.set_sensitive(True)
        else:
            self.minpixel_entry.set_sensitive(False)
            self.maxpixel_entry.set_sensitive(False)
            self.extent_entry.set_sensitive(False)
        if combobox.get_active_text() in ['azimuthal']:
            self.nphi_entry.set_sensitive(True)
        else:
            self.nphi_entry.set_sensitive(False)
        if combobox.get_active_text() in ['slices']:
            self.sectorwidth_entry.set_sensitive(True)
        else:
            self.sectorwidth_entry.set_sensitive(False)
        return True
    def set_beampos(self, coords):
        self.beamposx_entry.set_text(unicode(coords[0]))
        self.beamposy_entry.set_text(unicode(coords[1]))
