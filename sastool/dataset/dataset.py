# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:38:52 2011

@author: andris
"""
from arithmetic import ArithmeticBase
from errorvalue import ErrorValue
from attributealias import AliasedArrayAttributes

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.cbook import is_numlike
import scipy.optimize

class AliasedVectorAttributes(AliasedArrayAttributes, ArithmeticBase):
    """A specialization of AliasedArrayAttributes with the constraint that the
    attributes are (one-dimensional) vectors. The other constraint is, that it
    should have four aliased attributes, which can be unaliased to '_x', '_y',
    '_dx', '_dy'.
    
    These constraints are enough to define intuitive arithmetics with the help
    of the ArithmeticBase mixin. Currently right-hand sides can be:
        . instances of AliasedVectorAttributes (or of one of its subclasses)
        . instances of ErrorValue (or of one of its subclasses)
        . tuples of length at most 4 (treated as (x, y, dy, dx))
        . numeric entities (for which matplotlib.cbook.is_numlike() is True). In
            this case, the errors are assumed 0.
            
    __array__() is implemented, so np.array(obj) makes a structured numpy array
    from this.
    
    sorting is possible in-place with respect to any of the fields.
    
    Data contained in this can be saved into an ASCII file of the usual format
    (vectors in columns).
    
    sanitize() is defined to have a means to do various trimming and filtering
    operations.
    
    
    """
    _xtolerance = 1e-3
    def __init__(self, **kwargs):
        ArithmeticBase.__init__(self)
        kwargs['normalnames'] = ['x', 'y', 'dy', 'dx']
        AliasedArrayAttributes.__init__(self, **kwargs)
    def copy_into(self, into):
        """Helper function for copy(): make a deep copy,"""
        if not isinstance(into, AliasedVectorAttributes):
            raise TypeError('copy_into() cannot copy into other types than \
AliasedVectorAttributes or its subclasses')
        AliasedArrayAttributes.copy_into(self, into)
        into._xtolerance = self._xtolerance
    def _convert_numcompatible(self, c):
        """Convert c to a form usable by arithmetic operations"""
        #the compatible dataset to be returned, initialize it to zeros.
        comp = {'x':np.zeros_like(self._x),
              'y':np.zeros_like(self._x),
              'dy':np.zeros_like(self._x),
              'dx':np.zeros_like(self._x)}
        # if c is a DataSet:
        if isinstance(c, AliasedVectorAttributes):
            if self.shape() != c.shape(): # they are of incompatible size, fail.
                raise ValueError('incompatible length')
            # if the size of them is compatible, check if the abscissae are
            # compatible.
            xtol = min(self._xtolerance, c._xtolerance) # use the strictest
            if max(np.abs(self._x - c._x)) < xtol:
                try:
                    comp['x'] = c._x
                    comp['y'] = c._y
                    comp['dy'] = c._dy
                    comp['dx'] = c._dx
                except AttributeError:
                    pass # this is not a fatal error
            else:
                raise ValueError('incompatible abscissae')
        elif isinstance(c, ErrorValue):
            comp['x'] = self._x
            comp['y'] += c.val
            comp['dy'] += c.err
        elif isinstance(c, tuple): # if c is a tuple
            try:
                #the fields of comp were initialized to zero np arrays!
                comp['x'] += c[0]
                comp['y'] += c[1]
                comp['dy'] += c[2]
                comp['dx'] += c[3]
            except IndexError:
                pass # this is not fatal either
        else:
            if is_numlike(c):
                try:
                    comp['x'] = self._x
                    comp['y'] += c # leave this job to numpy.ndarray.__iadd__()
                except:
                    raise DataSetError('Incompatible size')
            else:
                raise DataSetError('Incompatible type')
        return comp
    def __iadd__(self, value):
        # self  += value operation
        try:
            comp = self._convert_numcompatible(value)
        except DataSetError:
            raise
        self._dx = 0.5 * np.sqrt(self._dx ** 2 + comp['dx'] ** 2)
        self._dy = np.sqrt(self._dy ** 2 + comp['dy'] ** 2)
        self._x = 0.5 * (self._x + comp['x'])
        self._y = self._y + comp['y']
        return self
    def __neg__(self):
        obj = self.copy()
        obj._y = -obj._y
        return obj
    def __imul__(self, value):
        # self  *= value operation
        try:
            comp = self._convert_numcompatible(value)
        except DataSetError:
            return NotImplemented
        self._dx = 0.5 * np.sqrt(self._dx ** 2 + comp['dx'] ** 2)
        self._dy = np.sqrt(self._dy ** 2 * comp['y'] ** 2 +
                                  self._y ** 2 * comp['dy'] ** 2)
        self._x = 0.5 * (self._x + comp['x'])
        self._y = self._y * comp['y']
        return self
    def _recip(self):
        """Calculate the reciprocal."""
        obj = self.copy()
        obj._dy = np.absolute(self._dy / self._y ** 2)
        return obj
    def sanitize(self, accordingto = None, thresholdmin = 0,
                 thresholdmax = np.inf, function = None, inplace = True,
                 finiteness = True, inclusive = False):
        """Do an in-place sanitization on this DataSet, i.e. remove nonfinite
        and out-of-bound elements.
        
        Inputs:
            accordingto: the field, which should be inspected, or a list of
                them. If None, defaults to self.fields(), i.e. all fields.
            thresholdmin: if the inspected field is smaller than this one, 
                the line is disregarded.
            thresholdmax: if the inspected field is larger than this one, 
                the line is disregarded.
            function: if this is not None, the validity of the dataline is
                decided from the boolean return value of function(value).
                Should accept a list and return a list of booleans. In this
                case, finiteness and the thresholds are NOT checked.
            inplace: if False, a copy will be created from the current object
                and it will be sanitized.
            finiteness: True if finiteness should be checked. False if not.
            inclusive: if function is None, include limits.
        """
        if accordingto is None:
            accordingto = '_y'
        if not (isinstance(accordingto, list) or
                isinstance(accordingto, tuple)):
            accordingto = [accordingto]
        indices = np.ones(self._shape, dtype = np.bool)
        for a in accordingto:
            a = self.unalias_keys(a)
            if hasattr(function, '__call__'):
                indices &= function(self.getfield(a))
            else:
                if finiteness:
                    indices &= np.isfinite(self.getfield(a))
                if inclusive:
                    indices &= ((self.getfield(a) >= thresholdmin) &
                              (self.getfield(a) <= thresholdmax))
                else:
                    indices &= ((self.getfield(a) > thresholdmin) &
                              (self.getfield(a) < thresholdmax))
        obj = self[indices]
        if inplace:
            obj.copy_into(self)
            return self
        else:
            return obj
    def trim(self, xmin = -np.inf, xmax = np.inf, inplace = False):
        """Trim the current dataset.
        
        Inputs:
            xmin: lower threshold in the abscissa
            xmax: upper threshold in the abscissa
            inplace [True]: if the original dataset should be trimmed or a copy
                of it
        Output: an instance of class DataSet with the elements of the original
            dataset where the abscissa is between xmin and xmax (limits
            included).
        """
        return self.sanitize(accordingto = '_x', thresholdmin = xmin,
                             thresholdmax = xmax, inplace = inplace,
                             finiteness = True, inclusive = True)
    def __array__(self, keys = None):
        """Make a structured numpy array from the current dataset.
        """
        if keys == None:
            keys = self.fields()
            values = self.fieldvalues()
        else:
            keys1 = self.unalias_keys(keys)
            values = [self.getfield(k) for k in keys1]
        a = np.array(zip(*values), dtype = zip(keys, [np.double] * len(keys)))
        return a

    def sort(self, order = '_x'):
        """Sort the current dataset according to 'order' (defaults to '_x').
        """
        order = self.unalias_keys(order)
        keys = self.unalias_keys(self.fields())
        a = self.__array__(keys)
        shrubbery = np.sort(a, order = order)
        for k in shrubbery.dtype.fields.keys():
            self.addfield(k, shrubbery[k], False)
        return self
    def save(self, filename, cols = None, formatstring = '%.16g'):
        """Save this dataset to an ascii file.
        
        Inputs:        
            filename: name of the file or a stream supporting write().
            cols: list of attributes to write. If None, '_x', '_y', '_dy', '_dx'
                are assumed. The first line of the file will contain the
                (aliased) column names.
            formatstring: a C-style format string to format numbers.
        """
        if cols is None:
            # default columns (may or may not exist)
            cols = ['_x', '_y', '_dy', '_dx']
            # other fields, which are not in cols, but we have them among the
            # fields.
            colsother = [x for x in self.unalias_keys(self.fields())
                        if not x in self.unalias_keys(cols)]
            cols.extend(colsother)
            # alias them...
            cols = self.alias_keys(cols)
            # remove fields which we do not have
            cols = [x for x in cols if self.unalias_keys(x) in
                    self.unalias_keys(self.fields())]
            # do not save fields starting with an underscore and having no
            # normal aliases.
            cols = [c for c in cols if not c.startswith('_')]
        # check if filename is a stream.
        if hasattr(filename, 'write'):
            f = filename
        else:
            # if not, open a file.
            f = open(filename, 'wt')
        # comment line of columns.
        f.write('#%s\n' % '\t'.join(cols))
        tmp = np.vstack([self.getfield(x) for x in cols]).T
        np.savetxt(f, tmp, fmt = formatstring)
        # format string to use for each line
        if not hasattr(filename, 'write'):
            # if we opened the file, close it.
            f.close()
    @classmethod
    def new_from_file(cls, filename, *args, **kwargs):
        """Load a 1D dataset from a file.
        
        Inputs:
            filename: the name of the file (or an open file-like object, which
                can be fed to np.loadtxt())
            All other positional and keyword arguments are forwarded to
                np.loadtxt()
        
        Output:
            a new instance of this class.
        """
        f = np.loadtxt(filename, *args, **kwargs) # this raises IOError if file cannot be loaded.
        N = f.shape[0]
        if N > 0:
            x = f[:, 0]
            dx = None
            dy = None
            if f.shape[1] == 1: # only one column, x will hold the row numbers
                y = f[:, 0]
                x = np.arange(f.shape[0], dtype = np.double)
            if f.shape[1] > 1:
                y = f[:, 1]
            if f.shape[1] > 2:
                dy = f[:, 2]
            if f.shape[1] > 3:
                dx = f[:, 3]
            return cls(x, y, dy, dx)
        else:
            raise ValueError('File %s does not contain any data points.' %
                                filename)
    def interpolate(self, newx):
        """Interpolate this dataset to a new abscissa (newx). Returns the new
        instance (the original is left intact)."""
        self1 = self.copy()
        self1.sort()
        obj = self.copy()
        # set shape to None, to allow validation of different shaped fields.
        obj._shape = None
        for k in self1.fields():
            obj.addfield(k, np.interp(newx, self1._x, self1.getfield(k)), False)
        obj._shape = obj.getfield(k).shape
        if hasattr(obj, 'validate'):
            obj.validate()
        return obj

    def momentum(self, exponent = 0, errorrequested = False):
        """Calculate momenta (integral from 0 to infinity of y times x^exponent)
        The integration is done by the trapezoid formula (np.trapz).
        
        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned (true Gaussian
                error-propagation of the trapezoid formula)
        """
        y = self._y * self._x ** exponent
        m = np.trapz(y, self._x)
        if errorrequested:
            err = self._dy * self._x ** exponent
            dm = errtrapz(self._x, err)
            return (m, dm)
        else:
            return m

    def extend(self, dataset):
        """Merge two datasets. The original is left intact, the merged one is 
        returned.
        
        Note that points with the same abscissa aren't treated specially, i.e.
        that point will exist in the abscissa of the merged curve _twice_!
        """
        obj = type(self)()
        obj._shape = None
        myfields = set(self.unalias_keys(self.fields()))
        otherfields = set(dataset.unalias_keys(dataset.fields()))
        common_fields = myfields.intersection(otherfields)
        obj.addfield('_x', np.concatenate((self._x, dataset._x)))
        for k in common_fields - set('_x'):
            obj.addfield(k, np.concatenate((self.getfield(k), dataset.getfield(k))))
        if hasattr(obj, 'validate'):
            obj.validate()
        return obj.sort()

    def unite(self, dataset, xmin = None, xmax = None, xsep = None, Npoints = 30,
              scaleother = True, verbose = True):
        """Merge 'dataset' with 'self' by scaling.
        
        Inputs:
            dataset: other dataset to unite this dataset with.
            xmin, xmax: overlap interval (for scaling) None to autodetect.
            xsep: separator in 'x'. The part of the current dataset before xsep
                and the part of the other dataset after xsep will be
                concatenated. If None, a simple merge (interleaved) is done.
            Npoints: number of points in the scaling interval (for re-binning)
            scaleother: scale the other dataset to this one. If false, the
                other way round.
            verbose: if the scaling integrals and the scaling factor is to
                be printed to the standard output
        
        Output: the merged dataset.
        
        Notes:
            1) for scaling, both datasets are interpolated to 
                np.linspace(xmin,xmax,Npoints). The scaling factor will be the 
                ratio of the integrals (by the trapezoid formula) of these two.
            2) the part of the current dataset BEFORE xsep will be added to
                the part of the other dataset AFTER xsep. If xsep is None, the
                two datasets will be merged simply.
        """
        if xmin is None:   #auto-determine
            xmin = max(self._x.min(), dataset._x.min())
        if xmax is None:   #auto-determine
            xmax = min(self._x.max(), dataset._x.max())
        if xmin > xmax:
            raise ValueError('Datasets do not overlap or xmin > xmax.')
        commonx = np.linspace(xmin, xmax, Npoints)
        selfint = self.interpolate(commonx)
        datasetint = dataset.interpolate(commonx)
        I1 = ErrorValue(*(selfint.momentum(errorrequested = True)))
        I2 = ErrorValue(*(datasetint.momentum(errorrequested = True)))
        obj = self.copy()
        if verbose:
            print "I1:", I1
            print "I2:", I2
            print "Uniting factor:", (I1 / I2)
        if scaleother:
            dataset = dataset * (I1 / I2)
        else:
            obj = obj * (I2 / I1)
        if xsep is not None:
            smallx = obj[obj._x <= xsep]
            bigx = dataset[dataset._x > xsep]
            if verbose:
                print "Small-x part: ", len(smallx), " data points."
                print "High-x part: ", len(bigx), " data points."
            uni = obj[obj._x <= xsep].extend(dataset[dataset._x > xsep])
        else:
            uni = obj.extend(dataset)
        if verbose:
            print "United dataset: ", len(uni), " data points."
        return uni
    @staticmethod
    def average(*datasets):
        """Average several datasets (weighted average, errors squared are the weights)
        """
        if len(datasets) == 1 and hasattr(datasets[0], '__getitem__'):
            datasets = datasets[0]
        res = datasets[0].copy()
        res._y = np.zeros_like(res._y)
        res._dy = np.zeros_like(res._y)
        for ds in datasets:
            w = 1 / ds._dy ** 2
            res._y += ds._y * w
            res._dy += w
        res._y /= res._dy
        res._dy = np.sqrt(1.0 / res._dy)
        return res



class MatrixAttrMixin(AliasedArrayAttributes, ArithmeticBase):
    _rowpos = 0
    _colpos = 0
    _rowpixsize = 1
    _colpixsize = 1
    def __init__(self, **kwargs):
        kwargs['normalnames'] = ['A', 'dA']
        AliasedArrayAttributes.__init__(self, **kwargs)
    def copy_into(self, into):
        if not isinstance(into, MatrixAttrMixin):
            raise TypeError('copy_into() cannot copy into other types than \
MatrixAttrMixin or its subclasses')
        AliasedArrayAttributes.copy_into(self, into)
        into._rowpos = self._rowpos
        into._colpos = self._colpos
        into._rowpixelsize = self._rowpixelsize
        into._colpixelsize = self._colpixelsize
    def _convert_numcompatible(self, c):
        """Convert c to a form usable by arithmetic operations"""
        #the compatible dataset to be returned, initialize it to zeros.
        if isinstance(c, MatrixAttrMixin):
            if self.shape() != c.shape(): # they are of incompatible size, fail.
                raise ValueError('incompatible shape')
            # if the size of them is compatible, check if the abscissae are
            # compatible.
            return (c._A, c._dA)
        elif isinstance(c, ErrorValue):
            return (c.val, c.err)
        elif isinstance(c, tuple): # if c is a tuple
            try:
                return (c[0], c[1])
            except IndexError:
                return (c[0], 0)
        else:
            if is_numlike(c):
                return c, 0
        raise DataSetError('Incompatible type')
    def __iadd__(self, rhs):
        val, err = self._convert_numcompatible(rhs)
        self._dA = np.sqrt(self._dA ** 2 + err ** 2)
        self._A = self._A + val
        return self
    def __imul__(self, rhs):
        val, err = self._convert_numcompatible(rhs)
        self._dA = np.sqrt(self._dA ** 2 * val ** 2 + self._A ** 2 * err ** 2)
        self._A = self._A * val
        return self
    def _recip(self):
        obj = self.copy()
        obj._A = 1. / self._A
        obj._dA = self._dA / (self._A * self._A)
        return obj
    def __neg__(self):
        obj = self.copy()
        obj._A = -self._A
        return obj
    def getXvec(self):
        return (np.arange(self.shape()[1]) - self._colpos) * self._colpixsize
    def getYvec(self):
        return (np.arange(self.shape()[0]) - self._rowpos) * self._rowpixsize
    def getXmat(self):
        return np.outer(np.ones(self.shape()[0]), self.getXvec)
    def getYmat(self):
        return np.outer(self.getYvec(), np.ones(self.shape()[1]))
    def getD(self):
        x = self.getXmat()
        y = self.getYmat()
        return np.sqrt(x ** 2 + y ** 2)


class PlotAndTransform(object):
    """This is a mixin class supporting plotting and transforming vectors. It
    should be mixed in alongside AliasedVectorAttributes. At least, the final
    class should have the attributes '_x', '_y', '_dx' ,'_dy' and addfield()."""
    _transform = None
    _plotaxes = None
    _plotuptodate = False
    def __init__(self,):
        object.__init__(self)
    def copy_into(self, into):
        """Helper function for copy(): make a deep copy."""
        if not isinstance(into, PlotAndTransform):
            raise TypeError('copy_into() cannot copy into other types than \
PlotAndTransform or its subclasses')
        into._plotaxes = self._plotaxes
        into.set_transform(self._transform)
    def set_transform(self, transform = None):
        """Set the transformation.
        'transform' should be an instance of Transform, or at least should have
        a 'name' attribute and be callable with the signature 
        d = transform(x, y, dy = None, dx = None) and d should be a dictionary
        with the transformed versions for x, y, dy, dx. This function makes
        the transformation implicitely."""
        self._transform = transform
        try:
            self._do_transform()
        except:
            self._invalidate_transform()
            raise
    def get_transform(self):
        """Return the current transform."""
        return self._transform
    def _invalidate_transform(self):
        """Invalidate the current transformation."""
        self._plotuptodate = False
        for key in ['_plotx', '_ploty', '_plotdx', '_plotdy']:
            self.removefield(key)
    def _do_transform(self):
        """Transform current dataset, make '_plotx' etc. attributes"""
        self._plotuptodate = False
        try:
            d = {'x':self._x, 'y':self._y, 'dy':self._dy, 'dx':self._dx}
        except AttributeError:
            # any of the fields are not defined. The cause for this may be the 
            # lack of _init_argument(). Do not tolerate this.
            raise
            #raise NotImplementedError('Fields cannot be found.')
        self._invalidate_transform()
        # if the transform is defined, do the transform. Otherwise leave d as
        # is.
        if self._transform is not None:
            d = self._transform(**d)
        #create the new fields.
        for k in d.keys():
            self.addfield('_plot%s' % k, d[k], False)
        self._plotuptodate = True
    def plot(self, *args, **kwargs):
        """Plot current dataset. Call plt.plot() with the appropriate arguments.
        """
        if not self._plotuptodate:
            self._do_transform()
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ax.plot(self._plotx, self._ploty, *args, **kwargs)
        self._plotaxes = ax
    def errorbar(self, *args, **kwargs):
        """Plot current dataset. Call plt.errorbar() with the appropriate
        arguments.
        """
        if not self._plotuptodate:
            self._do_transform()
        if '_plotdy' not in self.fields():
            dy = None
        else:
            dy = self._plotdy
        if '_plotdx' not in self.fields():
            dx = None
        else:
            dx = self._plotdx
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ax.errorbar(self._plotx, self._ploty, dy, dx, *args, **kwargs)
        self._plotaxes = ax
    def loglog(self, *args, **kwargs):
        """Plot current dataset. Call plt.loglog() with the appropriate
        arguments.
        """
        if not self._plotuptodate:
            self._do_transform()
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ax.loglog(self._plotx, self._ploty, *args, **kwargs)
        self._plotaxes = ax
    def semilogx(self, *args, **kwargs):
        """Plot current dataset. Call plt.semilogx() with the appropriate
        arguments.
        """
        if not self._plotuptodate:
            self._do_transform()
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ax.semilogx(self._plotx, self._ploty, *args, **kwargs)
        self._plotaxes = ax
    def semilogy(self, *args, **kwargs):
        """Plot current dataset. Call plt.semilogy() with the appropriate
        arguments.
        """
        if not self._plotuptodate:
            self._do_transform()
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ax.semilogy(self._plotx, self._ploty, *args, **kwargs)
        self._plotaxes = ax
    def _attr_validate(self, name, value):
        """Attribute validation plugin to invalidate plot when an attribute
        is added."""
        value = AliasedVectorAttributes._attr_validate(self, name, value)
        if self.unalias_keys(name) in ['_x', '_y', '_dx', '_dy']:
            try:
                assert(self.getfield(name) == value)
            except:
                self._do_transform()
        self._plotuptodate = False
        return value
    def trimzoomed(self, inplace = False, axes = None):
        """Trim dataset according to the current zoom on the last plot.
        
        Inputs:
            inplace: True if the current dataset is to be trimmed. If False, 
                a new instance of the same class is returned.
            axes: Use this axes instance.
                
        Notes:
            This method is useful to reduce the dataset to the currently viewed
                range. I.e. if this dataset was plotted using one of its plot
                methods (e.g. plot(), loglog(), errorbar(), ...) and the graph
                was zoomed in, then calling this function will eliminate the
                off-graph points.
            You will get undefined results if the axis has been deleted since
                the last plot of this instance.
        """
        if self._plotaxes is not None:
            axes = self._plotaxes
        if axes is None:
            raise ValueError('No plot axes corresponds to this dataset (and no \
overriding axes found)!')
        limits = axes.axis()
        obj = self.sanitize(accordingto = '_plotx', thresholdmin = limits[0],
                          thresholdmax = limits[1], inplace = inplace,
                          finiteness = True, inclusive = True)
        obj = obj.sanitize(accordingto = '_ploty', thresholdmin = limits[2],
                         thresholdmax = limits[3], inplace = inplace,
                         finiteness = True, inclusive = True)
        return obj
    def apparent(self):
        """Return an object of the apparent dataset."""
        obj = self.copy()
        obj.addfield('x', self._plotx, False)
        obj.addfield('y', self._ploty, False)
        obj.addfield('dx', self._plotdx, False)
        obj.addfield('dy', self._plotdy, False)
        obj.set_transform(None)
        return obj
    def validate(self):
        self._do_transform()

class DataSetError(StandardError):
    pass

class DataSet(AliasedVectorAttributes, PlotAndTransform):
    """ A general purpose dataset class. It has four special fields: x, y, 
        dx (sqrt(variance of x)), dy (sqrt(variance of y)), which can be
        accessed by object.fieldname and object['fieldname'] as well. Basic
        arithmetic operations are implemented with proper error propagation, 
        i.e. "a+b", "a/b" etc. yield the results expected normally. Other fields
        can also be added (but only via the object['newname'] = newfield method)
        which will compare the lengths of the new vector and object['x'] before
        the actual creation of the new field. However, be prepared that custom
        datafields may not be inherited by the results of arithmetic operations.
        Trimming and sanitization works, however.
        
        Plotting is also supported via the plot, semilogx, semilogy, loglog and
        errorbar methods. Optional arguments are forwarded to their matplotlib
        equivalents.
        
        Once plotted, the curves can be trimmed by trimzoomed().
        
        Saving to a text file and sanitization of the dataset are also
        supported.
        
        If you subclass this, you may be interested to redefine self._keytrans.
        This dictionary sets up aliases (even many levels deep) to datafields.
        By default, this is {'x':'_x', 'y':'_y', 'dy':'_dy', 'dx':'_dx'}, i.e
        the fields with underscores in their name (which are the ones actually
        stored in the object) can also be addressed under the names without
        underscores.
        
        Arithmetic operations are defined for the following pairs:
            class DataSet  < - >  scalar number
            class DataSet  < - >  class DataSet *
            class DataSet  < - >  numpy array **
            class DataSet  < - >  tuple ***
            
       *   only if (max_i(x1_i-x2_i) < min(xtolerance_1, xtolerance_2) is true
       **  only if the array is of the same length as field _x
       *** the tuple should contain at most 4 elements, in the sequence
              (x, y, dy, dx). Elements not present are assumed to be zero.
              
       Datasets can be converted to a structured numpy array by np.array().
       
       Different transformations can also be applied by the set_transform()
           method. These should be subclasses of Dataset.Transform, and only
           affect plotting and "trimzoomed".
    """
    _MCErrorPropSteps = 100
    def __init__(self, *args, **kwargs):
        """Initialize this instance.
        
        Example: DataSet([x [, y [, dy [, dx [, < keyword arguments > ]]]]])
        """
        PlotAndTransform.__init__(self)
        AliasedVectorAttributes.__init__(self, **kwargs)
        if len(args) > 4:
            raise TypeError('DataSet.__init__() takes at most 5 positional \
arguments (%d given, including \'self\')' % (len(args) + 1))
        for a, n in zip(args, self._normalnames):
            if n in kwargs.keys():
                raise TypeError('DataSet.__init__() got multiple values for \
keyword argument \'%s\'' % n)
            kwargs[n] = a
        for n in kwargs.keys():
            if self.couldhavefield(n):
                self.addfield(n, kwargs[n])
        if self.fields(): # don't call transform() if no fields are defined.
            self.set_transform()
    def _attr_validate(self, name, value):
        """Validator function to combine effects of the same function
        in both AliasedVectorAttributes and PlotAndTransform."""
        value = AliasedVectorAttributes._attr_validate(self, name, value)
        return PlotAndTransform._attr_validate(self, name, value)
    def copy_into(self, obj):
        AliasedVectorAttributes.copy_into(self, obj)
        PlotAndTransform.copy_into(self, obj)
        obj._MCErrorPropSteps = self._MCErrorPropSteps

    def evalfunction(self, function, *args, **kwargs):
        """Evaluate a function in the abscissa of the dataset.
        
        Inputs:
            function: a callable. First argument should be a numpy vector. All
                other optional arguments of DataSet.evalfunction() will be added
                as further arguments to this. Should return a numpy vector, same
                size as its first argument.
        
        Output: a DataSet where the abscissa is the same as that of the original
            DataSet. The ordinate will be the evaluated function. Error
            propagation will also be calculated if the error of the abscissa is
            not zero, by a Monte Carlo process (number of iterations: 
            _MCErrorPropSteps attribute of the DataSet instance).
        """
        x = self._x
        y = function(x, *args, **kwargs)
        if (self._MCErrorPropSteps > 1) and (self._dx.sum() > 0):
            dy = np.zeros_like(x)
            for i in xrange(self._MCErrorPropSteps):
                dy += (y - function(x + self._dx * np.random.randn(len(x)))) ** 2
            dy = np.sqrt(dy) / (self._MCErrorPropSteps - 1)
            dx = self._dx
        else:
            dy = None
            dx = None
        ret = self.__class__(x, y, dy, dx)
        ret.set_transform(self._transform)
        return ret

    def plotfitted(self, function, params, dparams = None, chi2 = None,
                   dof = None, funcinfo = None):
        """Plot a nice graph from the results of a fitting.
        
        Inputs:
            function: fitting function (callable), see method evalfunction() for
                details.
            params: fitted parameters (list)
            dparams: errors of fitted parameters (list, None to skip)
            chi2: reduced chi-squared (None to skip)
            dof: degrees of freedom (None to skip)
            funcinfo: dictionary containing information on the fit function. Use
                None to skip this.
                Available fields (any of these can be omitted):
                    'funcname' (string): name of the function
                    'paramnames' (list of strings): names of function parameters
                    'plotmethod' (string): name of a member function of class
                        DataSet which should be used for plotting (e.g. 'plot', 
                        'loglog', 'semilogy', 'errorbar', ...)
                    'formula' (string): formula of the function
                    'logtext' (list of callables): each element will be called
                        as l(d, params, dparams, chi2, dof) where d is a
                        dictionary containing '_x', '_y', '_dy', '_dx' of the
                        original dataset at least. Other fields may be added in
                        later versions. These functions should return strings,
                        which will be appended to the log box
                        
        Outputs: Nothing

        A graph will be drawn to the current axes (plt.gca()). The original
            curve will be drawn with blue markers and no line, the fitted one
            with red line and no markers. A log box similar to the
            "Fitting results" in Origin(R) will be overlaid on the axes.
        """
        if funcinfo is not None:
            #make a copy so we can update it without destroying the original.
            funcinfo = funcinfo.copy()
        else:
            funcinfo = {}
            funcinfo['funcname'] = 'Model function'
            funcinfo['paramnames'] = ['parameter #%d' % (i + 1) for i in
                                    range(len(params))]
        cfitted = self.evalfunction(function, *params)
        if 'plotmethod' not in funcinfo.keys():
            funcinfo['plotmethod'] = 'plot'
        if dparams is None:
            dparams = [np.nan] * len(params)
        self.__getattribute__(funcinfo['plotmethod']).__call__(linestyle = ' ',
                                                               marker = '.',
                                                               color = 'b')
        cfitted.__getattribute__(funcinfo['plotmethod'])(linestyle = '-',
                                                         marker = '',
                                                         color = 'r')
        logtext = u"Function: %s\n" % funcinfo['funcname']
        if 'formula' in funcinfo.keys():
            logtext = u"Formula: %s\n" % funcinfo['formula']
        logtext += u"Parameters:\n"
        for i in range(len(params)):
            if dparams is None:
                logtext += u"    %s : %g \n" % (funcinfo['paramnames'][i],
                                              params[i])
            else:
                logtext += u"    %s : %g +/- %g\n" % (funcinfo['paramnames'][i],
                                                    params[i], dparams[i])
        if chi2 is not None:
            logtext += u"Reduced chi^2: %g\n" % chi2
        if dof is not None:
            logtext += u"Degrees of freedom: %d\n" % dof
        if 'logtext' in funcinfo.keys():
            for i in funcinfo['logtext']:
                logtext += u"%s" % i.__call__({'_x':self._x, '_y':self._y,
                                             '_dy':self._dy, '_dx':self._dx},
                                            params, dparams, chi2, dof)
        plt.text(0.95, 0.95, logtext, bbox = {'facecolor':'white', 'alpha':0.6,
                                         'edgecolor':'black'},
                 ha = 'right', va = 'top', multialignment = 'left',
                 transform = plt.gca().transAxes)

    def fit(self, function, parinit = None, funcinfo = {}, doplot = True,
            ext_output = False, **kwargs):
        """Perform a least-squares fit to the dataset.
        
        Inputs:
            function: a callable or an instance of fitfunction.FitFunction.
                Should be able to be called as function(x, param1, param2, ...)
                and should return a numpy array of the same size as x.
            parinit: 1) if 'function' is an instance of fitfunction.Fitfunction, 
                        this can be None. Otherwise, it can be:
                     2) a callable: will be called as parinit(x, y) and should
                         return a list of initial (approximated) values for all
                         fitting parameters needed by 'function', or
                     3) an iterable (np vector or a list or a tuple etc.)
                         containing the initial values of all the fitting
                         parameters.
            funcinfo: only needed if doplot is True. Then it should be a dict, 
                which will be forwarded to the member function plotfitted(), 
                after updating with function.funcinfo if function was an
                instance of FitFunction.
            ext_output: if extended output is needed (see outputs)
            All other keyword arguments are passed to scipy.optimize.leastsq()
        Outputs: p, pstd, [infodict]
            p: list of fitted parameters
            pstd: list of errors of fitted parameters
            infodict (only if 'ext_output' was true): a dictionary of other
                parameters. It holds every parameter returned by the call to
                scipy.optimize.leastsq in the output argument of the same name, 
                extended with the following (currently):
                    'Chi2': reduced Chi^2
                    'dof': degrees of freedom
                    'mesg': string message from scipy.optimize.leastsq()
                    'ier': integer status from scipy.optimize.leastsq()
                    'weighting': weigthing method, see Notes.
                    'cov_x': estimated covariance matrix of the parameters
                    'R2': R^2 parameter
        
        Notes:
            for the fitting itself, scipy.optimize.leastsq() is used. Weighting
            mode is determined from the dataset:
                1) if the dx vector is zero, uniform weighting is used.
                2) if some (but not all) elements of the dx vector are zeros, 
                    those elements of the dataset are used only for fitting
                    (this corresponds to infinite weights to those points) with
                    uniform weights
                3) otherwise instrumental weighting is used, i.e. w = 1/dy**2
                        
        """
        funcinfo = funcinfo.copy() # thus we can update it freely
        if hasattr(function, 'funcinfo'):
            funcinfo.update(function.funcinfo())
        if parinit is None and hasattr(function, 'init_arguments'):
            parinit = function.init_arguments(self._x, self._y)
        #get the initial parameters
        if hasattr(parinit, '__call__'):
            params_initial = parinit(self._x, self._y)
        else:
            params_initial = parinit
        x = self._x
        y = self._y
        if (self._dy.sum() == 0):
            w = np.ones_like(self._x)
            weighting = 'uniform'
        elif (self._dy == 0).sum() > 0:
            idx = self._dy == 0
            x = x[idx]
            y = y[idx]
            w = np.ones_like(x)
            weighting = 'uniform, only points where sigma = 0'
        else:
            w = 1 / self._dy
            weighting = 'instrumental'
        def func(p, x, y, w, f = function):
            return (f(x, *(p.tolist())) - y) * w
        p, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(func,
                                                               params_initial,
                                                               args = (x, y, w),
                                                               full_output = 1,
                                                               **kwargs)
#        if ier > 4 or ier < 1:
#            raise DataSetError(mesg)
        chisquare = (infodict['fvec'] ** 2).sum()
        degrees_of_freedom = len(self) - len(p)
        if cov_x is None:
            pstd = [np.nan] * len(p)
        else:
            pstd = [ np.sqrt(cov_x[i, i] * chisquare / degrees_of_freedom) for i in
                            range(len(p))]
        sserr = np.sum(((function(x, *(p.tolist())) - y) * w) ** 2)
        sstot = np.sum((y - np.mean(y)) ** 2 * w ** 2)
        r2 = 1 - sserr / sstot
        if funcinfo is not None and doplot:
            self.plotfitted(function, p, pstd, chisquare, degrees_of_freedom,
                            funcinfo)
        if ext_output:
            infodict['Chi2'] = chisquare
            infodict['dof'] = degrees_of_freedom
            infodict['mesg'] = mesg
            infodict['ier'] = ier
            infodict['weighting'] = weighting
            if cov_x is None:
                infodict['cov_x'] = np.zeros((len(p), len(p))) * np.nan
            else:
                infodict['cov_x'] = np.array(cov_x) * float(chisquare) / float(degrees_of_freedom)
            infodict['R2'] = r2
            return p, pstd, infodict
        else:
            return p, pstd


class SASCurve(DataSet):
    def __init__(self, *args, **kwargs):
        if 'keytrans' not in kwargs.keys():
            kwargs['keytrans'] = {}
        kwargs['keytrans'].update({'q':'_x', 'Intensity':'_y', 'Error':'_dy',
                                   'dq':'_dx'})
        DataSet.__init__(self, *args, **kwargs)

def errtrapz(x, yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable
        
    Outputs:
        the error of the integral
    """
    x = np.array(x)
    yerr = np.array(yerr)
    return 0.5 * np.sqrt((x[1] - x[0]) ** 2 * yerr[0] ** 2 +
                        np.sum((x[2:] - x[:-2]) ** 2 * yerr[1:-1] ** 2) +
                        (x[-1] - x[-2]) ** 2 * yerr[-1] ** 2)


