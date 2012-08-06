import gtk
import matplotlib
from ... import misc

from PyGTKCallback import PyGTKCallback
@PyGTKCallback
class PlotTab(gtk.HBox):
    lastplotraw = None
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

        frame = gtk.Frame()
        self.pack_start(frame, False)
        tab = gtk.Table()
        frame.add(tab)

        tablecolumn = 0;
        tablerow = 0;

        l = gtk.Label('Color scale:'); l.set_alignment(0, 0.5);
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.colorscale_combo = gtk.combo_box_new_text()
        tab.attach(self.colorscale_combo, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.colorscale_combo.append_text('linear')
        self.colorscale_combo.append_text('log')
        self.colorscale_combo.append_text('sqrt')
        self.colorscale_combo.set_active(1)
        self.colorscale_combo.connect('changed', self.on_params_changed)

        tablerow += 1
        l = gtk.Label('Palette:'); l.set_alignment(0, 0.5);
        tab.attach(l, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.palette_combo = gtk.combo_box_new_text()
        tab.attach(self.palette_combo, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        idx = 0
        indexfound = 0
        defaultpalette = misc.sastoolrc.get('gui.sasimagegui.plot.palette')
        for cm in sorted(dir(matplotlib.cm), key=lambda x:x.upper()):
            if eval('isinstance(getattr(matplotlib.cm,cm),matplotlib.colors.Colormap)') and not cm.endswith('_r'):
                self.palette_combo.append_text(cm)
                if cm == defaultpalette.rsplit('_', 1)[0]:
                    indexfound = idx
                idx += 1
        self.palette_combo.set_active(indexfound)
        self.palette_combo.connect('changed', self.on_params_changed)

        tablerow += 1
        self.reversepalette_checkbutton = gtk.CheckButton('Reverse palette?');
        self.reversepalette_checkbutton.set_alignment(0, 0.5);
        tab.attach(self.reversepalette_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.reversepalette_checkbutton.set_active(defaultpalette.endswith('_r'))
        self.reversepalette_checkbutton.connect('toggled', self.on_params_changed)

        tablecolumn += 1
        tablerow = 0
        self.plotqrange_checkbutton = gtk.CheckButton('Q values on axes?')
        self.plotqrange_checkbutton.set_alignment(0, 0.5)
        tab.attach(self.plotqrange_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.plotqrange_checkbutton.set_active(True)
        self.plotqrange_checkbutton.connect('toggled', self.on_params_changed)

        tablerow += 1
        self.plotmask_checkbutton = gtk.CheckButton('Plot mask?')
        self.plotmask_checkbutton.set_alignment(0, 0.5)
        tab.attach(self.plotmask_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.plotmask_checkbutton.set_active(True)
        self.plotmask_checkbutton.connect('toggled', self.on_params_changed)

        tablerow += 1
        self.crosshair_checkbutton = gtk.CheckButton('Beam position?')
        self.crosshair_checkbutton.set_alignment(0, 0.5)
        tab.attach(self.crosshair_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.crosshair_checkbutton.set_active(True)
        self.crosshair_checkbutton.connect('toggled', self.on_params_changed)

        tablecolumn += 1
        tablerow = 0
        self.drawcolorbar_checkbutton = gtk.CheckButton('Draw colorbar?')
        self.drawcolorbar_checkbutton.set_alignment(0, 0.5)
        tab.attach(self.drawcolorbar_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 2, tablerow, tablerow + 1)
        self.drawcolorbar_checkbutton.set_active(True)
        self.drawcolorbar_checkbutton.connect('toggled', self.on_params_changed)

        tablerow += 1
        self.lowclip_checkbutton = gtk.CheckButton('Lower clip val.:'); self.lowclip_checkbutton.set_alignment(0, 0.5);
        tab.attach(self.lowclip_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.lowclip_checkbutton.connect('toggled', self.on_clip_toggled)
        self.lowclip_entry = gtk.Entry()
        tab.attach(self.lowclip_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        tablerow += 1
        self.upperclip_checkbutton = gtk.CheckButton('Upper clip val.:'); self.upperclip_checkbutton.set_alignment(0, 0.5);
        tab.attach(self.upperclip_checkbutton, 2 * tablecolumn, 2 * tablecolumn + 1, tablerow, tablerow + 1, gtk.FILL, gtk.FILL)
        self.upperclip_checkbutton.connect('toggled', self.on_clip_toggled)
        self.upperclip_entry = gtk.Entry()
        tab.attach(self.upperclip_entry, 2 * tablecolumn + 1, 2 * tablecolumn + 2, tablerow, tablerow + 1)

        self.lowclip_checkbutton.set_active(False)
        self.upperclip_checkbutton.set_active(False)
        self.on_clip_toggled(None)

    def on_button_clicked(self, button, arg):
        self.emit(arg)
    def on_params_changed(self, widget):
        cscale = self.palette_combo.get_active_text()
        if self.reversepalette_checkbutton.get_active():
            cscale = cscale + '_r'
        misc.sastoolrc.set('gui.sasimagegui.plot.palette', cscale)
        self.emit('plotparams-changed')
    def on_clip_toggled(self, widget):
        self.upperclip_entry.set_sensitive(self.upperclip_checkbutton.get_active())
        self.lowclip_entry.set_sensitive(self.lowclip_checkbutton.get_active())
    def update_from_data(self, data):
        if data is None:
            return False
        if not data.check_for_mask(False):
            self.plotmask_checkbutton.set_active(False)
        plotmask = self.plotmask_checkbutton.get_active()
        if len(data.check_for_q(False)) > 0:
            self.plotqrange_checkbutton.set_active(False)
        if not self.upperclip_checkbutton.get_active():
            self.upperclip_entry.set_text(unicode(float(data.max(plotmask))))
        if not self.lowclip_checkbutton.get_active():
            self.lowclip_entry.set_text(unicode(float(data.min(plotmask))))
        return True
    def plot2d(self, data, axes):
        if data is None:
            return False
        self.update_from_data(data)
        colorscale = self.colorscale_combo.get_active_text()
        palette = self.palette_combo.get_active_text()
        if self.reversepalette_checkbutton.get_active():
            palette = palette + '_r'

        if self.drawcolorbar_checkbutton.get_active():
            if len(axes.figure.axes) > 1:
                drawcolorbar = axes.figure.axes[1]
            else:
                drawcolorbar = True
        else:
            drawcolorbar = False
        try:
            minvalue = float(self.lowclip_entry.get_text())
        except ValueError:
            self.emit('error', 'Invalid (non-numeric) min.value: ' + self.lowclip_entry.get_text())
            minvalue = float(data.min(self.plotmask_checkbutton.get_active()))
        try:
            maxvalue = float(self.upperclip_entry.get_text())
        except ValueError:
            self.emit('error', 'Invalid (non-numeric) max.value: ' + self.upperclip_entry.get_text())
            maxvalue = float(data.max(self.plotmask_checkbutton.get_active()))

        ret, self.lastplotraw = data.plot2d(axes=axes, cmap=getattr(matplotlib.cm, palette),
                                         zscale=colorscale,
                                         drawmask=self.plotmask_checkbutton.get_active(),
                                         qrange_on_axis=self.plotqrange_checkbutton.get_active(),
                                         crosshair=self.crosshair_checkbutton.get_active(),
                                         drawcolorbar=drawcolorbar,
                                         minvalue=minvalue,
                                         maxvalue=maxvalue,
                                         return_matrix=True)
        return True
    def get_axesunits(self):
        if self.plotqrange_checkbutton.get_active():
            return 'q'
        else:
            return 'pixel'
