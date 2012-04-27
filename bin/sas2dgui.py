#!/usr/bin/python

import matplotlib
matplotlib.use('GTKAgg')
import sastool.gui
import gtk
a=sastool.gui.saspreview2d.SAS2DGUI()
def delete_handler(*args,**kwargs):
  gtk.main_quit()
a.connect('delete-event',delete_handler)
a.show_all()
gtk.main()
