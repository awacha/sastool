# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:21:02 2011

@author: -
"""
import numpy as np

class AliasedAttributes(object):
    """Mixin class to allow for aliased and validated attributes. Aliasing is 
    accomplished via a key translation dictionary (keytrans), where keytrans[x]
    is the name to which x is an alias. This dictionary can be defined in the
    constructor. A validation function _attr_validate() can also be defined. 
    This will be called as value_corrected = self._attr_validate(name, value) and
    should check if 'value' is valid for field 'name'. It should try to convert
    'value' in a compatible form and return it, or raise an exception otherwise.
    
    A typical usage usually means subclassing this and overriding the functions
    _attr_validate() and copy_into().
    """
    _keytrans = {}
    _dict = {}
    def __init__(self, **kwargs):
        object.__init__(self) # call the upstream constructor
        # create defaults for arguments 'keytrans' and 'dict_'
        if 'keytrans' not in kwargs.keys():
            kwargs['keytrans'] = {}
        if 'dict_' not in kwargs.keys():
            kwargs['dict_'] = {}
        # if the arguments 'keytrans' and 'dict_' are dictionaries, copy them to
        # fields self._dict and self._keytrans. Otherwise raise exceptions.
        if isinstance(kwargs['keytrans'], dict):
            self._keytrans = kwargs['keytrans'].copy()
        else:
            raise ValueError('Argument "keytrans" should be a dictionary!')
        if isinstance(kwargs['dict_'], dict):
            self._dict = kwargs['dict_'].copy()
        else:
            raise ValueError('Argument "dict_" should be a dictionary!')
    def addfield(self, name, value, validate = True):
        """self.addfield(name, value, validate): add a new field. The field
            will be accessible as self.'name'. Set validate = False to suppress
            automatic calling of self._attr_validate() on the new value (use
            with care, this can break consistency).
        
        For example:
            a.addfield('q', [1, 2, 3]')
            a.q   - >   [1, 2, 3]
        """
        #these hacks are needed because this can be called from _inti_attribute.
        selfdict = object.__getattribute__(self, '_dict')
        if hasattr(self,'_attr_validate'):
            sav = object.__getattribute__(self, '_attr_validate')
        else:
            validate = False
        name = self.unalias_keys(name) # so we don't need to bother with aliases
        if validate:
            value = sav(name, value)
        selfdict[name] = value
    def __getattr__(self, key):
        #Note that this function gets called only if self.__getattribute__
        # failed.
        #Take aliasing into account. All attributes are stored internally under
        # their unaliased names.
        key1 = self.unalias_keys(key)
        #this is needed to avoid infinite loops
        selfdict = object.__getattribute__(self, '_dict')
        #Try to find the key in self._dict
        if key1 in selfdict.keys():
            return selfdict[key1]
        #otherwise try to initialize it, if we have self._init_attribute defined
        elif hasattr(self,'_init_attribute'):
            try:
                # again, to avoid infinite loops.
                return object.__getattribute__(self,'_init_attribute')(key)
            except NotImplementedError:
                # if self._init_attribute raises NotImplementedError, we have to
                # raise an AttributeError, which is done if the end of this
                # function is reached.
                pass
        # We reached this point without being able to resolve 'key': raise the
        # appropriate exception.
        raise AttributeError(key)
    def __delattr__(self, key):
        # try to delete an attribute. First check if this is an aliased
        # attribute. If not, call the upstream __delattr__
        key = self.unalias_keys(key)
        if key in self._dict.keys():
            del self._dict[key]
        object.__delattr__(self, key)
    def __setattr__(self, key, value):
        #Resolution order:
        # 1) check if the unaliased version of the key exists in self._dict. If
        #    it does, set that field by calling self.addfield() and return
        # 2) call the upstream __setattr__.
        key1 = self.unalias_keys(key)
        if key1 in self._dict.keys() or key1 in self._keytrans.values():
            self.addfield(key, value)
        else:
            object.__setattr__(self, key, value)
    def alias_keys(self, k):
        """Find the topmost alias for all keys in argument."""
        # as this function can be called from __getattr__ or from functions
        # called by that, we have to take extra care.
        if isinstance(k, list):
            # vectorize
            return [object.__getattribute__(self,'alias_keys')(x) for x in k]
        else:
            skt = object.__getattribute__(self, '_keytrans')
            while (k in skt.values()): # loop to control nested aliases
                # if multiple aliases exist, take the first one.
                k = [x for x in skt.keys() if skt[x] == k][0]
            return k
    def unalias_keys(self, k):
        """Unalias all keys in argument (recursively)."""
        # as this function can be called from __getattr__ or from functions
        # called by that, we have to take extra care.
        if isinstance(k, list):
            return [object.__getattribute__(self,'unalias_keys')(x) for x in k]
        else:
            skt = object.__getattribute__(self, '_keytrans')
            while(k in skt.keys()): # loop to control nested aliases
                k = skt[k]
            return k
    def fieldvalues(self):
        """Get current values for all defined aliased attributes."""
        return self._dict.values()
    def copy(self):
        """Make a deep copy by using self.copy_into(). This is a class-
        independent function, i.e. subclasses can use this without further
        modifications. The return type is the class from which it was called.
        """
        obj = self.__class__()
        self.copy_into(obj)
        return obj
    def copy_into(self, into):
        """Helper for copy(). Ensures that relevant attributes are copied into
        'into'. This function should be overridden by subclasses. But take care
        to call this function from inside the new!"""
        if not isinstance(into, AliasedAttributes):
            raise TypeError('copy_into() cannot copy into other types than \
AliasedAttributes or its subclasses')
        for k in self._dict.keys():
            if hasattr(self._dict[k], 'copy'):
                into._dict[k] = self._dict[k].copy()
            elif isinstance(self._dict[k], list):
                into._dict[k] = self._dict[k][:]
            else:
                into._dict[k] = self._dict[k]
        into._keytrans.update(self._keytrans)
    def fields(self):
        """Return the (aliased) names of the currently defined fields."""
        return self.alias_keys(self._dict.keys())
    def hasfield(self, key):
        """Check if 'key' is a valid fieldname (aliasing is accounted for)"""
        return (self.alias_keys(key) in self.fields())
    def removefield(self, key):
        """Remove field 'key' (aliasing is accounted for)"""
        if isinstance(key, list):
            return [self.removefield(k) for k in key]
        else:
            key = self.unalias_keys(key)
            if key in self._dict.keys():
                return self._dict.__delitem__(key)
            else:
                return None
    def couldhavefield(self, key):
        """Check if this instance could have a field 'key' (i.e. it is among the
        keys defined by the key translation dictionary)."""
        if isinstance(key, list):
            return [object.__getattribute__(self,'couldhavefield')(k) for k in
                    key]
        else:
            return (object.__getattribute__(self,'unalias_keys')(key) in 
                    object.__getattribute__(self,'_keytrans').values())
    def clear(self):
        """Remove all aliased fields"""
        self.removefield(self.fields())
    def asdict(self):
        """Return a dictionary representation."""
        d={}
        for k in self.fields():
            d[k]=self.getfield(k)
        return d
    # shortcut
    getfield = __getattr__

class AliasedArrayAttributes(AliasedAttributes):
    """This mixin class adds support for aliased _array_ attributes. It inherits
    from AliasedAttributes to perform attribute aliasing. All aliased attributes
    need to be numpy ndarrays of the same shape. This is ensured by a validating
    function _attr_validate().
    
    The attribute initialization mechanism of AliasedAttributes is used to
    initialize not available attributes to zero if they are referenced. 
    
    The constructor accepts 'normalnames', a keyword argument. It should be a
    list defining the names of the aliased attributes. Their unaliased
    counterparts will be constructed by prepending an underscore to their names.
    The keyword argument 'keytrans' will be constructed appropriately for
    AliasedAttributes. If 'keytrans' was supplied to the constructor, it will be
    updated accordingly.
    
    Slicing and array-like indexing is implemented (only getting, but not
    setting or deleting slices).
    """
    def __init__(self, **kwargs):
        if 'normalnames' not in kwargs.keys():
            kwargs['normalnames'] = []
        self._normalnames = kwargs['normalnames']
        del kwargs['normalnames']
        self._protnames = ['_%s'%a for a in self._normalnames]
        if 'keytrans' not in kwargs.keys():
            kwargs['keytrans'] = {}
        
        #update 'keytrans'
        kwargs['keytrans'].update(dict(zip(self._normalnames, self._protnames)))
        AliasedAttributes.__init__(self, **kwargs)
        self._shape = None
    def _init_attribute(self,name):
        #this is called by AliasedAttributes.__getattr__ if 'name' is not found.
        
        #if '_shape' is None or an empty list, we cannot do anything, raise an
        # exception
        shape=object.__getattribute__(self, '_shape')
        if not shape: 
            raise NotImplementedError

        #If 'name' is not a field name we know of, we cannot do anything.
        if not object.__getattribute__(self,'couldhavefield')(name):
            raise NotImplementedError
        #otherwise set the field to zero.
        object.__getattribute__(self, 'addfield')(name, np.zeros(shape), False)
    def copy_into(self, into):
        """Helper function for copy(): deep copying."""
        if not isinstance(into, AliasedArrayAttributes):
            raise TypeError('copy_into() cannot copy into other types than \
AliasedArrayAttributes or its subclasses')
        AliasedAttributes.copy_into(self, into)
        into._shape = self._shape
        into._protnames = self._protnames[:]
        into._normalnames = self._normalnames[:]
    def shape(self):
        """Return the (common) shape of the attributes"""
        return self._shape
    def __len__(self):
        if self._shape is not None:
            return reduce(lambda a, b:a*b, [x for x in self._shape if x > 0])
        else:
            return None
    def _attr_validate(self, name, value):
        """Validator function"""
        if value is None:
            if self._shape is None:
                raise ValueError('None is not allowed for the first field in \
DataSet!')
            else:
                # make it zero
                value = np.zeros(self._shape)
        value = np.array(value) # convert it to a numpy array
        if not self._shape: # None or an empty tuple/list
            self._shape = value.shape
        if value.shape != self._shape:
            raise ValueError('Invalid shape for new field!')
        return value
    def __getitem__(self, key):
        """Slicing. Forward arguments to the __getitem__ members of the aliased
        attributes (of type numpy.ndarray)."""
        #make a copy
        obj = self.copy()
        # set shape to None, to allow validation of different shaped fields.
        obj._shape = None
        for k in self.fields():
            obj.addfield(k, self.getfield(k)[key], False)
        obj._shape = obj.getfield(k).shape
        return obj
    def clear(self):
        """Clear all attributes."""
        AliasedAttributes.clear(self)
        self._shape = None
