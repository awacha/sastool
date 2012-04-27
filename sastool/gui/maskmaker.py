'''
Created on Feb 14, 2012

@author: andris
'''

from ..io.twodim import readmask

import gtk
import matplotlib
matplotlib.use('GtkAgg')
import matplotlib.backends
if matplotlib.backends.backend.upper() != 'GTKAGG':
    raise ImportError('Sastool.gui works only with the GTK backend of \
Matplotlib')

import uuid
import numpy as np
import matplotlib.nxutils
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg, \
NavigationToolbar2GTKAgg
from matplotlib.figure import Figure
import os
import scipy.io

#Mask matrix should be plotted with plt.imshow(maskmatrix, cmap=_colormap_for_mask)
_colormap_for_mask=matplotlib.colors.ListedColormap(['white','white'],'_sastool_gui_saspreview2d_maskcolormap')
_colormap_for_mask._init()
_colormap_for_mask._lut[:,-1]=0
_colormap_for_mask._lut[0,-1]=0.7



class StatusLine(gtk.HBox):
    _signal_handlers = []
    _button_clicks = []

    def __init__(self, Nbuttons=2):
        gtk.HBox.__init__(self)
        self.label = gtk.Label()
        self.label.set_alignment(0, 0)
        self.pack_start(self.label)
        self.button = []
        for i in range(Nbuttons):
            self.button.append(gtk.Button())
            self.pack_start(self.button[i], False, True)
            self.button[i].connect('clicked', self.buttonclicked)
        self._button_clicks = [0] * Nbuttons
        self.label.show()
        self.connect('expose_event', self.on_expose)

    def buttonclicked(self, widget):
        i = self.button.index(widget)
        self._button_clicks[i] += 1
        for f, a, kw in self._signal_handlers:
            f(self, i, *a, **kw)

    def connect(self, eventname, function, *args, **kwargs):
        if eventname == 'buttonclicked':
            hid = uuid.uuid4()
            self._signal_handlers.append((hid, function, args, kwargs))
        else:
            return gtk.HBox.connect(self, eventname, function, *args, **kwargs)

    def disconnect(self, hid):
        self._signal_handlers = [x for x in self._signal_handlers \
                                 if not x[0] == hid]

    def setup(self, text=None, *args):
        if text is None:
        #    self.hide()
            text = ''
        #else:
        #    self.show()
        self.label.set_text(text)
        for b in self.button:
            b.set_label('')
            b.hide()
        for b, t in zip(self.button, args):
            if t is not None:
                b.set_label(t)
                b.show()

    def clear(self):
        self.setup('')
        self.hide()

    def nbuttonclicks(self, i=None):
        if i is None:
            ret = self._button_clicks[:]
            self._button_clicks = [0] * len(self._button_clicks)
        else:
            ret = self._button_clicks[i]
            self._button_clicks[i] = 0
        return ret

    def on_expose(self, *args):
        for b in self.button:
            b.set_visible(bool(b.get_label()))

    def reset_counters(self, n=None):
        if n is None:
            self._button_clicks = [0] * len(self._button_clicks)
        else:
            self._button_clicks[n] = 0


class GraphToolbarVisibility(object):

    def __init__(self, toolbar, *args):
        self.graphtoolbar = toolbar
        self.widgetstohide = args
        self._was_zooming = False
        self._was_panning = False

    def __enter__(self):
        if self.graphtoolbar.mode.startswith('zoom'):
            self.graphtoolbar.zoom()
            self._was_zooming = True
        else:
            self._was_zooming = False
        if self.graphtoolbar.mode.startswith('pan'):
            self.graphtoolbar.pan()
            self._was_panning = True
        else:
            self._was_panning = False
        self.graphtoolbar.set_sensitive(False)
        for w in self.widgetstohide:
            w.set_sensitive(False)
        while gtk.events_pending():
            gtk.main_iteration()

    def __exit__(self, *args, **kwargs):
        if self._was_panning:
            self.graphtoolbar.pan()
        if self._was_zooming:
            self.graphtoolbar.zoom()
        self.graphtoolbar.set_sensitive(True)
        for w in self.widgetstohide:
            w.set_sensitive(True)
        while gtk.events_pending():
            gtk.main_iteration()


class MaskMaker(gtk.Dialog):
    _mouseclick_mode = None  # Allowed: 'Points', 'Lines', 'PixelHunt' and None
    _mouseclick_last = ()
    _selection = None

    def __init__(self, title='Make mask...', parent=None,
                 flags=gtk.DIALOG_DESTROY_WITH_PARENT | \
                 gtk.DIALOG_NO_SEPARATOR,
                 buttons=(gtk.STOCK_OK, gtk.RESPONSE_OK, gtk.STOCK_CANCEL,
                          gtk.RESPONSE_CANCEL),
                 matrix=None, mask=None):
        if matrix is None:
            raise ValueError("Argument 'matrix' is required!")
        gtk.Dialog.__init__(self, title, parent, flags, buttons)
        self._matrix = matrix
        if mask is None:
            mask = np.ones_like(matrix).astype(np.bool8)
        self._mask = mask.copy()
        self.set_default_response(gtk.RESPONSE_CANCEL)

        clearbutton = gtk.Button(stock = gtk.STOCK_NEW)
        self.action_area.pack_end(clearbutton)
        clearbutton.connect('clicked', self.newmask)
        clearbutton.show()
        savebutton = gtk.Button(stock = gtk.STOCK_SAVE_AS)
        self.action_area.pack_end(savebutton)
        savebutton.connect('clicked', self.savemask)
        savebutton.show()
        loadbutton = gtk.Button(stock = gtk.STOCK_OPEN)
        self.action_area.pack_end(loadbutton)
        loadbutton.connect('clicked', self.loadmask)
        loadbutton.show()

        hbox = gtk.HBox()
        self.vbox.pack_start(hbox)
        self.toolbar = gtk.VBox()
        hbox.pack_start(self.toolbar, False, True)

        figvbox = gtk.VBox()
        hbox.pack_start(figvbox)
        self.fig = Figure(figsize = (0.5, 0.4), dpi = 72)
        self.fig.add_subplot(111)

        self.canvas = FigureCanvasGTKAgg(self.fig)
        self.canvas.set_size_request(300, 200)
        figvbox.pack_start(self.canvas, True, True, 0)
        self.canvas.mpl_connect('button_press_event', self._on_matplotlib_mouseclick)

        hb1 = gtk.HBox()
        figvbox.pack_start(hb1, False, True, 0)
        self.graphtoolbar = NavigationToolbar2GTKAgg(self.canvas, self.vbox)
        hb1.pack_start(self.graphtoolbar, True, True, 0)
        #assemble toolbar on the left
        self.selector_buttons = gtk.VBox()
        self.toolbar.pack_start(self.selector_buttons)
        self.masking_buttons = gtk.VBox()
        self.toolbar.pack_start(self.masking_buttons)
        for name, func, container in [('Pixel hunt', self.pixelhunt, self.selector_buttons),
                          ('Select rectangle', self.selectrect, self.selector_buttons),
                          ('Select circle', self.selectcircle, self.selector_buttons),
                          ('Select polygon', self.selectpoly, self.selector_buttons),
                          ('Select by histogram\n(not yet masked pixels)', self.selecthisto_notyetmasked, self.selector_buttons),
                          ('Select by histogram\n(from all pixels)', self.selecthisto, self.selector_buttons),
                          ('Forget selection', self.clearselection, self.masking_buttons),
                          ('Mask selection', self.maskselection, self.masking_buttons),
                          ('Unmask selection', self.unmaskselection, self.masking_buttons),
                          ('Flip mask\nover selection', self.flipmaskselection, self.masking_buttons),
                          ('Flip mask', self.flipmask, self.selector_buttons),
                          ('Mask nonfinite pixels', self.masknonfinite, self.selector_buttons),
                          ('Mask nonpositive pixels', self.masknonpositive, self.selector_buttons),
                          ]:
            b = gtk.Button(name)
            container.pack_start(b)
            b.connect('clicked', func)

        self.statusline = StatusLine()
        self.statusline.setup(None)
        self.vbox.pack_start(self.statusline, False, True, 0)

        self.vbox.show_all()
        self.update_graph(True)
        self.set_select_mode(True)
    def get_mask(self):
        return self._mask.copy()
    def update_graph(self, redraw = False):
        if redraw:
            self.fig.clf()
        im = self.fig.gca().imshow(self._matrix, interpolation = 'nearest')
        self.fig.gca().imshow(self._mask, cmap=_colormap_for_mask, interpolation = 'nearest')
        if redraw:
            self.fig.colorbar(im)
        self.canvas.draw()
    def newmask(self, widget):
        self._mask = np.ones_like(self._matrix).astype(np.bool8)
        self.update_graph(True)
        return True
    def savemask(self, widget):
        fcd = gtk.FileChooserDialog('Select file to save mask...', self,
                                  gtk.FILE_CHOOSER_ACTION_SAVE,
                                  (gtk.STOCK_OK, gtk.RESPONSE_OK,
                                   gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL))
        fcd.set_current_folder(os.getcwd())
        fcd.set_destroy_with_parent(True)
        fcd.set_modal(True)
        ff = gtk.FileFilter(); ff.set_name('All files'); ff.add_pattern('*');
        fcd.add_filter(ff)
        ff = gtk.FileFilter(); ff.set_name('Matlab(R) mask matrices'); ff.add_pattern('*.mat');
        fcd.add_filter(ff)
        fcd.set_filter(ff)
        if fcd.run() == gtk.RESPONSE_OK:
            filename = fcd.get_filename()
            scipy.io.savemat(filename, {'mask':self._mask})
        os.chdir(fcd.get_current_folder())
        fcd.destroy()
    def loadmask(self, widget):
        fcd = gtk.FileChooserDialog('Select file to load mask...', self,
                                  gtk.FILE_CHOOSER_ACTION_OPEN,
                                  (gtk.STOCK_OK, gtk.RESPONSE_OK,
                                   gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL))
        fcd.set_current_folder(os.getcwd())
        fcd.set_destroy_with_parent(True)
        fcd.set_modal(True)
        ff = gtk.FileFilter(); ff.set_name('All files'); ff.add_pattern('*');
        fcd.add_filter(ff)
        ff = gtk.FileFilter(); ff.set_name('Matlab(R) mask matrices'); ff.add_pattern('*.mat');
        fcd.add_filter(ff)
        fcd.set_filter(ff)
        if fcd.run() == gtk.RESPONSE_OK:
            filename = fcd.get_filename()
            try:
                mask1 = readmask(filename).astype(np.bool8)
            except:
                self.statusline.setup('Invalid mask file.')
            else:
                if mask1.shape != self._mask.shape:
                    self.statusline.setup('Incompatible mask shape.')
                else:
                    self._mask = mask1
        os.chdir(fcd.get_current_folder())
        fcd.destroy()
        self.update_graph(True)
        return True
    def set_select_mode(self, value = True):
        self.selector_buttons.set_sensitive(value)
        self.masking_buttons.set_sensitive(not value)
        return True
    def _on_matplotlib_mouseclick(self, event):
        if self._mouseclick_mode is None:
            return False
        if event.button == 1:
            if self._mouseclick_mode.upper() == 'POINTS':
                ax = self.fig.gca().axis()
                self.fig.gca().plot(event.xdata, event.ydata, 'o', c = 'white', markersize = 7);
                self.fig.gca().axis(ax)
                self.fig.canvas.draw();
            if self._mouseclick_mode.upper() == 'LINES':
                ax = self.fig.gca().axis()
                self.fig.gca().plot(event.xdata, event.ydata, 'o', c = 'white', markersize = 7)
                if self._mouseclick_last:
                    self.fig.gca().plot([self._mouseclick_last[-1][0], event.xdata],
                                        [self._mouseclick_last[-1][1], event.ydata],
                                        c = 'white');
                self.fig.gca().axis(ax)
                self.fig.canvas.draw();
            if self._mouseclick_mode.upper() == 'PIXELHUNT':
                if (event.xdata >= 0 and event.xdata < self._mask.shape[1] and
                    event.ydata >= 0 and event.ydata < self._mask.shape[0]):
                    self._mask[round(event.ydata), round(event.xdata)] ^= 1;
                    self.update_graph()
            self._mouseclick_last.append((event.xdata, event.ydata));
    def pixelhunt(self, widget):
        with GraphToolbarVisibility(self.graphtoolbar, self.toolbar):
            self.statusline.setup('Click pixels to change masking. If finished, press --->', 'Finished')
            self.statusline.reset_counters()
            self._mouseclick_last = []
            self._mouseclick_mode = 'Pixelhunt'
            while not self.statusline.nbuttonclicks(0):
                gtk.main_iteration()
            self._mouseclick_mode = None
        self.statusline.setup(None)
        self.update_graph()
        self.set_select_mode()
        return True
    def masknonfinite(self, widget):
        self._mask &= np.isfinite(self._matrix)
        self.update_graph()
        return True
    def masknonpositive(self, widget):
        self._mask &= (self._matrix > 0)
        self.update_graph()
        return True
    def selecthisto(self, widget, data = None):
        self.toolbar.set_sensitive(False)
        if data is None:
            data = self._matrix
        self._mouseclick_mode = None
        self.fig.clf()
        Nbins = max(min(data.max() - data.min(), 100), 1000)
        self.fig.gca().hist(data.flatten(), bins = Nbins, normed = True)
        self.fig.canvas.draw()
        self.statusline.setup('Zoom range to select, then press ---->', 'Finished')
        self.statusline.reset_counters()
        while not self.statusline.nbuttonclicks(0):
            gtk.main_iteration()
        ax = self.fig.gca().axis()
        self._selection = (self._matrix >= ax[0]) & (self._matrix <= ax[1])
        self.statusline.setup('Range %g<=data<=%g selected: %u pixels.' % (ax[0], ax[1], self._selection.sum()))
        self.update_graph(True)
        self.toolbar.set_sensitive(True)
        self.set_select_mode(False)
        return True
    def selecthisto_notyetmasked(self, widget):
        return self.selecthisto(widget, data = self._matrix[self._mask.astype('bool')])
    def selectrect(self, widget):
        with GraphToolbarVisibility(self.graphtoolbar, self.toolbar):
            self.statusline.setup('Click two opposite corners of the rectangle')
            self._mouseclick_last = []
            self._mouseclick_mode = 'POINTS'
            while len(self._mouseclick_last) < 2:
                gtk.main_iteration()
            self._mouseclick_mode = None
            ax = self.fig.gca().axis()
            x0 = min([t[0] for t in self._mouseclick_last])
            x1 = max([t[0] for t in self._mouseclick_last])
            y0 = min([t[1] for t in self._mouseclick_last])
            y1 = max([t[1] for t in self._mouseclick_last])
            self.fig.gca().plot([x0, x0, x1, x1, x0],
                                [y0, y1, y1, y0, y0],
                                c = 'white')
            self.fig.gca().axis(ax)
            col, row = np.meshgrid(range(self._mask.shape[1]),
                                range(self._mask.shape[0]))
            self._selection = (row >= y0) & (row <= y1) & (col >= x0) & (col <= x1)
            self.fig.canvas.draw()
        self.statusline.setup(None)
        self.set_select_mode(False)
        return True
    def selectcircle(self, widget):
        with GraphToolbarVisibility(self.graphtoolbar, self.toolbar):
            self.statusline.setup('Click the center of the circle')
            self._mouseclick_last = []
            self._mouseclick_mode = 'POINTS'
            while len(self._mouseclick_last) < 1:
                gtk.main_iteration()
            self.statusline.setup('Click a peripheric point of the circle')
            while len(self._mouseclick_last) < 2:
                gtk.main_iteration()
            self._mouseclick_mode = None
            ax = self.fig.gca().axis()
            xcen = self._mouseclick_last[0][0]
            ycen = self._mouseclick_last[0][1]
            r = np.sqrt((self._mouseclick_last[1][0] - xcen) ** 2 +
                      (self._mouseclick_last[1][1] - ycen) ** 2)
            t = np.linspace(0, np.pi * 2, 1000);
            self.fig.gca().plot(xcen + r * np.cos(t), ycen + r * np.sin(t),
                                c = 'white')
            self.fig.gca().axis(ax)
            col, row = np.meshgrid(range(self._mask.shape[1]),
                                range(self._mask.shape[0]))
            self._selection = ((row - ycen) ** 2 + (col - xcen) ** 2 <= r ** 2)
            self.fig.canvas.draw()
        self.statusline.setup(None)
        self.set_select_mode(False)
        return True
    def selectpoly(self, widget):
        with GraphToolbarVisibility(self.graphtoolbar, self.toolbar):
            self.statusline.setup('Select corners of the polygon. If finished, press --->', 'Finished')
            self.statusline.reset_counters()
            self._mouseclick_last = []
            self._mouseclick_mode = 'LINES'
            while not self.statusline.nbuttonclicks(0):
                gtk.main_iteration()
            self._mouseclick_mode = None
            if len(self._mouseclick_last) > 2:
                ax = self.fig.gca().axis()
                self.fig.gca().plot([self._mouseclick_last[-1][0],
                                     self._mouseclick_last[0][0]],
                                    [self._mouseclick_last[-1][1],
                                     self._mouseclick_last[0][1]],
                                    c = 'white')
                self.fig.gca().axis(ax)
                self.fig.canvas.draw()
                col, row = np.meshgrid(range(self._mask.shape[1]),
                                    range(self._mask.shape[0]))
                points = np.vstack((col.flatten(), row.flatten())).T
                Nrows, Ncols = self._mask.shape
                points_inside = matplotlib.nxutils.points_inside_poly(points, self._mouseclick_last)
                self._selection = np.zeros((Nrows, Ncols), np.bool8)
                self._selection[points_inside.astype('bool').reshape((Nrows, Ncols))] = 1
        self.statusline.setup(None)
        self.set_select_mode(False)
        return True
    def clearselection(self, widget):
        self._selection = None
        self.update_graph(redraw = True)
        self.set_select_mode()
        return True
    def maskselection(self, widget):
        if self._selection is None:
            return False
        self._mask &= -self._selection
        self.update_graph(redraw = True)
        self._selection = None
        self.set_select_mode()
        return True
    def unmaskselection(self, widget):
        if self._selection is None: return False
        self._mask |= self._selection
        self.update_graph(redraw = True)
        self._selection = None
        self.set_select_mode()
        return True
    def flipmaskselection(self, widget):
        if self._selection is None: return False
        self._mask[self._selection] ^= 1
        self.update_graph(redraw = True)
        self._selection = None
        self.set_select_mode()
        return True
    def flipmask(self, widget):
        self._mask ^= 1;
        self.update_graph(redraw = True)
        return True

def makemask(matrix = None, mask0 = None):
    mm = MaskMaker(matrix = matrix, mask = mask0)
    resp = mm.run()
    if resp == gtk.RESPONSE_OK:
        mask0 = mm.get_mask()
    mm.destroy()
    return mask0
