import uuid

class PyGTKCallback(object):
    def __init__(self, cls):
        self._wrappedcls = cls
    def __call__(self, *args, **kwargs):
        obj = self._wrappedcls(*args, **kwargs)
        obj._pygtkcallbacks = []
        def connect(name, callbackfunc, *args):
            try:
                retval = self._wrappedcls.connect(obj, name, callbackfunc, *args)
            except TypeError as te:
                if 'unknown signal name' in te.message:
                    if hasattr(obj, '_valid_pygtk_signalnames'):
                        if name not in obj._valid_pygtk_signalnames:
                            raise TypeError(repr(obj) + ": unknown signal name: " + name)
                    obj._pygtkcallbacks.insert(0, {'slotobject':obj, 'name':name, 'id':uuid.uuid1(), 'callback':callbackfunc, 'args':args, 'block':0})
                    retval = obj._pygtkcallbacks[-1]['id']
                else:
                    raise
            return retval
        def connect_after(name, callbackfunc, *args):
            try:
                retval = self._wrappedcls.connect_after(obj, name, callbackfunc, *args)
            except TypeError as te:
                if 'unknown signal name' in te.message:
                    if hasattr(obj, '_valid_pygtk_signalnames'):
                        if name not in obj._valid_pygtk_signalnames:
                            raise TypeError(repr(obj) + ": unknown signal name: " + name)
                    obj._pygtkcallbacks.append({'slotobject':obj, 'name':name, 'id':uuid.uuid1(), 'callback':callbackfunc, 'args':args, 'block':0})
                    retval = obj._pygtkcallbacks[-1]['id']
                else:
                    raise
            return retval
        def connect_object(name, callbackfunc, slotobject, *args):
            try:
                retval = self._wrappedcls.connect_object(obj, name, callbackfunc, slotobject, *args)
            except TypeError as te:
                if 'unknown signal name' in te.message:
                    if hasattr(obj, '_valid_pygtk_signalnames'):
                        if name not in obj._valid_pygtk_signalnames:
                            raise TypeError(repr(obj) + ": unknown signal name: " + name)
                    obj._pygtkcallbacks.insert(0, {'slotobject':slotobject, 'name':name, 'id':uuid.uuid1(), 'callback':callbackfunc, 'args':args, 'block':0})
                    retval = obj._pygtkcallbacks[-1]['id']
                else:
                    raise
            return retval
        def connect_object_after(name, callbackfunc, slotobject, *args):
            try:
                retval = self._wrappedcls.connect_object_after(obj, name, callbackfunc, slotobject, *args)
            except TypeError as te:
                if 'unknown signal name' in te.message:
                    if hasattr(obj, '_valid_pygtk_signalnames'):
                        if name not in obj._valid_pygtk_signalnames:
                            raise TypeError(repr(obj) + ": unknown signal name: " + name)
                    obj._pygtkcallbacks.append({'slotobject':slotobject, 'name':name, 'id':uuid.uuid1(), 'callback':callbackfunc, 'args':args, 'block':0})
                    retval = obj._pygtkcallbacks[-1]['id']
                else:
                    raise
            return retval
        def emit(name, *args):
            try:
                self._wrappedcls.emit(obj, name, *args)
            except TypeError as te:
                if 'unknown signal name' in te.message:
                    for p in obj._pygtkcallbacks:
                        if p['name'] == name and p['block'] == 0:
                            p['callback'](p['slotobject'], *(args + p['args']))
                else:
                    raise
        def disconnect(handler_id):
            if isinstance(handler_id, uuid.UUID):
                obj._pygtkcallbacks = [x for x in obj._pygtkcallbacks if x['id'] != handler_id]
            else:
                self._wrappedcls.disconnect(obj, handler_id)
        def handler_block(handler_id):
            if isinstance(handler_id, uuid.UUID):
                for p in obj._pygtkcallbacks:
                    if p['id'] == id:
                        p['block'] += 1
            else:
                self._wrappedcls.handler_block(handler_id)
        def handler_unblock(handler_id):
            if isinstance(handler_id, uuid.UUID):
                for p in obj._pygtkcallbacks:
                    if p['id'] == id and p['block'] > 0:
                        p['block'] -= 1
            else:
                self._wrappedcls.handler_unblock(handler_id)
        def handler_is_connected(handler_id):
            if isinstance(handler_id, uuid.UUID):
                return bool([p for p in obj._pygtkcallbacks if p['id'] == id])
            return self._wrappedcls.handler_is_connected(handler_id)
        obj.connect = connect
        obj.connect_after = connect_after
        obj.connect_object = connect_object
        obj.connect_object_after = connect_object_after
        obj.disconnect = disconnect
        obj.emit = emit
        obj.handler_block = handler_block
        obj.handler_unblock = handler_unblock
        obj.handler_is_connected = handler_is_connected
        return obj
