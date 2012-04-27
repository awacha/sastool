'''
Created on Apr 19, 2012

@author: andris
'''

import gtk
import gobject
import os

from .. import misc

class PathEditor(gtk.Dialog):
    def __init__(self,parent=None):
        if parent is not None:
            parent=parent.get_toplevel()
        gtk.Dialog.__init__(self,'Edit sastool search path...',parent,
                            gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                             gtk.STOCK_OK, gtk.RESPONSE_OK))
        self.set_default_response(gtk.RESPONSE_CANCEL)
        hbox=gtk.HBox()
        self.vbox.pack_start(hbox)
        
        sw=gtk.ScrolledWindow()
        hbox.pack_start(sw)
        
        self.pathstore=gtk.ListStore(gobject.TYPE_STRING)
        self.tw=gtk.TreeView(self.pathstore)
        sw.add(self.tw)
        self.tw.set_size_request(300,150);
        self.tw.set_grid_lines(gtk.TREE_VIEW_GRID_LINES_HORIZONTAL)
        self.tw.set_rules_hint(True)
        self.tw.set_reorderable(True)
        self.tw.set_enable_search(True)
        
        
        pathcolumn=gtk.TreeViewColumn('Folder',gtk.CellRendererText(),text=0)
        self.tw.append_column(pathcolumn)
        
        bb=gtk.VButtonBox()
        hbox.pack_start(bb,False)
        b=gtk.Button(label='Add folder')
        b.connect('clicked',self.callback_add)
        bb.add(b)
        b=gtk.Button(label='Add current folder')
        b.connect('clicked',self.callback_add,'.')
        bb.add(b)
        b=gtk.Button(label='Add home folder')
        b.connect('clicked',self.callback_add,os.path.expanduser('~'))
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_GOTO_TOP)
        b.connect('clicked',self.callback_move,'top')
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_GO_UP)
        b.connect('clicked',self.callback_move,'up')
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_GO_DOWN)
        b.connect('clicked',self.callback_move,'down')
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_GOTO_BOTTOM)
        b.connect('clicked',self.callback_move,'bottom')
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_REMOVE)
        b.connect('clicked',self.callback_remove)
        bb.add(b)
        b=gtk.Button(stock=gtk.STOCK_CLEAR)
        b.connect('clicked',self.callback_clear)
        bb.add(b)
        
        
        self.update_from_search_path()
        self.show_all()
        self.hide()
    def run(self,*args,**kwargs):
        self.update_from_search_path()
        return gtk.Dialog.run(self,*args,**kwargs)
    def update_from_search_path(self):
        self.pathstore.clear()
        for k in misc.get_search_path():
            self.pathstore.append([k])
    def update_search_path(self):
        it=self.pathstore.get_iter_first()
        mypath=[]
        while it is not None:
            mypath.append(self.pathstore.get_value(it,0))
            it=self.pathstore.iter_next(it)
        misc.set_search_path(mypath)
    def callback_move(self,button,whatmove):
        it=self.tw.get_selection().get_selected()[1]
        if it is None:
            return
        if whatmove=='down':
            nextit=self.pathstore.iter_next(it)
            if nextit is not None:
                self.pathstore.move_after(it,nextit)
        elif whatmove=='up':
            nr=self.pathstore.get_path(it)[0]
            nr_to=max(nr-1,0)
            self.pathstore.move_before(it,self.pathstore.get_iter(nr_to))
        elif whatmove=='top':
            self.pathstore.move_after(it,None)
        elif whatmove=='bottom':
            self.pathstore.move_before(it,None)
        else:
            assert True
    def callback_remove(self,button=None):
        it=self.tw.get_selection().get_selected()[1]
        if it is None:
            return
        nr=self.pathstore.get_path(it)[0]
        self.pathstore.remove(it)
        if len(self.pathstore):
            self.tw.get_selection().select_iter(self.pathstore.get_iter(min(nr,len(self.pathstore)-1)))
    def callback_clear(self,button=None):
        self.pathstore.clear()
    def callback_add(self,button=None,folder=None):
        if folder is not None:
            self.pathstore.prepend([folder])
            return True
        if not hasattr(self,'_filechooser_for_add'):
            self._filechooser_for_add=gtk.FileChooserDialog(title='Choose folder...',
                parent=self,action=gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
                buttons=(gtk.STOCK_OK, gtk.RESPONSE_OK, gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL))
            self._filechooser_for_add.set_modal(True)
            self._filechooser_for_add.set_transient_for(self)
            self._filechooser_for_add.set_destroy_with_parent(True)
            self._filechooser_for_add.withsubfolders=gtk.CheckButton('Recursive add')
            self._filechooser_for_add.withsubfolders.set_active(True)
            self._filechooser_for_add.set_extra_widget(self._filechooser_for_add.withsubfolders)
        if self._filechooser_for_add.run()==gtk.RESPONSE_OK:
            folder=self._filechooser_for_add.get_filename()
            if self._filechooser_for_add.withsubfolders.get_active():
                os.path.walk(folder,lambda x,y,z:self.pathstore.prepend([y]),None)
            else:
                self.pathstore.prepend([folder])
        self._filechooser_for_add.hide()

def pathedit(mainwindow=None):
    pe=PathEditor(mainwindow)
    if pe.run()==gtk.RESPONSE_OK:
        pe.update_search_path()
    pe.destroy()
