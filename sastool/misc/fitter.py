import numbers
import time

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares, OptimizeResult

from .cormap import cormaptest

__all__ = ['Fitter']

class Fitter:
    def __init__(self, function, parameters, x, y, dx=None, dy=None, lbounds=None, ubounds=None, ytransform=None,
                 loss=None,
                 method=None):
        """Class representing a nonlinear least-squares fitting of a dataset

        Inputs:
            function: function
                the model function to perform the fit with. Expected signature:
                func(x, a1, a2, ...) where x is the independent variable, must
                handle double precision scalars or np.ndarrays. The return value
                should be the same type (and shape) as x. The other parameters
                are the model parameters to be fitted.
            parameters: list of floats and FixedParameter
                the initial parameters. If the type is FixedParameter, the
                parameter will be kept as constant during the fit.
            x, y, dx, dy: 1D np.ndarrays of floating point numbers
                the dataset against the fitting must be carried out.
            lbounds, ubounds: list or None
                List of lower and upper bounds for the corresponding parameters.
                None means unbounded. If only a subset of parameters are to
                be constrained, you can give -np.inf and np.inf to the others.
            ytransform: str, currently 'ln' is supported
                If the y data (and the fitting function) must be transformed
                before starting the fit.
            loss: The loss function, see scipy.optimize.least_squares
            method: fit method, see scipy.optimize.least_squares

        In order to carry out the fitting, call the fit() method.
        """
        self.ytransform = ytransform
        self._function = function
        self._parameters = parameters
        self._fixed_parameters = [None] * len(parameters)
        self._covariance = np.zeros((len(parameters), len(parameters)))
        self._uncertainties = [0] * len(self._parameters)
        self._loss = loss
        self._method = method
        self.setX(x)
        self.setY(y)
        self.setDX(dx)
        self.setDY(dy)
        self.setLbounds(lbounds)
        self.setUbounds(ubounds)
        self._stats = {}

    def freeParameters(self):
        return [p for p, fp in zip(self._parameters, self._fixed_parameters) if fp is None]

    def freeLbounds(self):
        return [lb for lb, fp in zip(self._lbounds, self._fixed_parameters) if fp is None]

    def freeUbounds(self):
        return [ub for ub, fp in zip(self._ubounds, self._fixed_parameters) if fp is None]

    def checkBounds(self):
        return all([lb <= p <= ub for p, lb, ub in zip(self.freeParameters(), self.freeLbounds(), self.freeUbounds())])

    def fit(self, loss=None, method=None):
        starttime = time.monotonic()
        kwargs = {}
        if not self.checkBounds():
            return self
        if loss is None:
            loss = self._loss
        if method is None:
            method = self._method
        if loss is not None:
            kwargs['loss'] = loss
        if method is not None:
            kwargs['method'] = method
        result = least_squares(
            self._fitfunction(), self.freeParameters(),
            bounds=(self.freeLbounds(), self.freeUbounds()), **kwargs)
        assert isinstance(result, OptimizeResult)
        self._stats = {'success': result.success, 'result': result, 'message': result.message, 'status': result.status,
                       'time': time.monotonic() - starttime, 'nfev': result.nfev, 'njev': result.njev,
                       'cost': result.cost,
                       'active_mask_original': result.active_mask, 'optimality': result.optimality, 'grad': result.grad,
                       'jac': result.jac, 'num_func_eval': result.nfev, 'error_flag': result.status, }
        # if not result.success:
        #    return self
        _, s, VT = svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        self._parameters = self._substitute_fixed_parameters(result.x.tolist())
        self._covariance = self._substitute_fixed_parameters_covar(pcov)
        self._uncertainties = np.diag(self._covariance, 0) ** 0.5
        funcvalue = self.evaluateFunction()
        active_mask_resolved = self._fixed_parameters[:]
        active_mask_orig = result.active_mask.tolist()
        for i in range(len(active_mask_resolved)):
            if active_mask_resolved[i] is None:
                active_mask_resolved[i] = active_mask_orig.pop(0)
            else:
                active_mask_resolved[i] = 0
        self._stats['Covariance'] = self._covariance
        self._stats['Correlation_coeffs'] = self._covariance / np.outer(self._uncertainties, self._uncertainties)
        self._stats['func_value'] = funcvalue
        self._stats['active_mask'] = active_mask_resolved
        self._stats['Chi2'] = (result.fun ** 2).sum()
        self._stats['DoF'] = len(self.x()) - len(self.freeParameters())
        self._stats['Chi2_reduced'] = self._stats['Chi2'] / self._stats['DoF']
        self._stats['SStot'] = ((self.y() - np.mean(self.y())) ** 2).sum()
        self._stats['SSres'] = ((funcvalue - self.y()) ** 2).sum()
        self._stats['R2'] = 1 - self._stats['SSres'] / self._stats['SStot']
        self._stats['R2_adj'] = 1 - (self._stats['SSres'] / (len(self.x()) - len(self.freeParameters()) - 1)) / (
        self._stats['SStot'] / (len(self.x()) - 1))
        if self.dy() is not None:
            self._stats['SStot_weighted'] = (((self.y() - np.mean(self.y())) / self.dy()) ** 2).sum()
            self._stats['SSres_weighted'] = (((funcvalue - self.y()) / self.dy()) ** 2).sum()
            self._stats['R2_weighted'] = 1 - self._stats['SSres_weighted'] / self._stats['SStot_weighted']
            self._stats['R2_adj_weighted'] = 1 - (self._stats['SSres_weighted'] / (
            len(self.x()) - len(self.freeParameters()) - 1)) / (self._stats['SStot_weighted'] / (len(self.x()) - 1))
        else:
            for key in ['SStot', 'SSres', 'R2', 'R2_adj']:
                self._stats[key + '_weighted'] = self._stats[key]
        self._stats['CorMapTest_p'], self._stats['CorMapTest_C'], self._stats['CorMapTest_CorMap'] = cormaptest(
            self.y(), self._stats['func_value'])
        self._stats['CorMapTest_n'] = len(self.y())
        return self

    def success(self):
        return self._stats['success']

    def stats(self):
        return self._stats

    def setX(self, x):
        self._x = x

    def setY(self, y):
        self._y = y

    def setDX(self, dx):
        self._dx = dx

    def setDY(self, dy):
        self._dy = dy

    def function(self):
        return self._function

    def parameters(self):
        return self._parameters

    def setParameters(self, parameters):
        self._parameters = parameters

    def uncertainties(self):
        return np.diag(self._covariance, 0) ** 0.5

    def lbounds(self):
        return self._lbounds

    def ubounds(self):
        return self._ubounds

    def setLbounds(self, lbounds):
        if lbounds is None:
            lbounds = [-np.inf] * len(self.parameters())
        if isinstance(lbounds, numbers.Number):
            lbounds = [lbounds] * len(self.parameters())
        self._lbounds = lbounds

    def setUbounds(self, ubounds):
        if ubounds is None:
            ubounds = [np.inf] * len(self.parameters())
        if isinstance(ubounds, numbers.Number):
            ubounds = [ubounds] * len(self.parameters())
        self._ubounds = ubounds

    def x(self):
        return self._x

    def y(self):
        return self._y

    def dx(self):
        return self._dx

    def dy(self):
        return self._dy

    def _substitute_fixed_parameters(self, args):
        parameters = self._fixed_parameters[:]
        assert len(args) == parameters.count(None)
        for a in args:
            parameters[parameters.index(None)] = a
        return parameters

    def _substitute_fixed_parameters_covar(self, covar):
        """Insert fixed parameters in a covariance matrix"""
        covar_resolved = np.empty((len(self._fixed_parameters), len(self._fixed_parameters)))
        indices_of_fixed_parameters = [i for i in range(len(self.parameters())) if
                                       self._fixed_parameters[i] is not None]
        indices_of_free_parameters = [i for i in range(len(self.parameters())) if self._fixed_parameters[i] is None]
        for i in range(covar_resolved.shape[0]):
            if i in indices_of_fixed_parameters:
                # the i-eth argument was fixed. This means that the row and column corresponding to this argument
                # must be None
                covar_resolved[i, :] = 0
                continue
            for j in range(covar_resolved.shape[1]):
                if j in indices_of_fixed_parameters:
                    covar_resolved[:, j] = 0
                    continue
                covar_resolved[i, j] = covar[indices_of_free_parameters.index(i), indices_of_free_parameters.index(j)]
        return covar_resolved

    def fixparameters(self, arglist=None):
        if arglist is None:
            self._fixed_parameters = [None] * len(self.parameters())
        else:
            assert len(arglist) == len(self.parameters())
            assert all([isinstance(a, (type(None), numbers.Number)) for a in arglist])
            self._fixed_parameters = list(arglist)

    def unfixparameters(self):
        return self.fixparameters()

    def _fitfunction(self):
        x = self.x()
        y = self.y()
        dy = self.dy()
        if self.ytransform == 'ln':
            idx = y > 0
            if dy is not None:
                dy = dy[idx] / np.abs(y[idx])
            y = np.log(y[idx])
            x = x[idx]
            if dy is None:
                def func(args):
                    return y - np.log(self._function(x, *self._substitute_fixed_parameters(args.tolist())))
            else:
                def func(args):
                    return (y - np.log(self._function(x, *self._substitute_fixed_parameters(args.tolist())))) / dy
        else:
            if dy is None:
                def func(args):
                    return y - self._function(x, *self._substitute_fixed_parameters(args.tolist()))
            else:
                def func(args):
                    return (y - self._function(x, *self._substitute_fixed_parameters(args.tolist()))) / dy
        return func

    def evaluateFunction(self):
        return self._function(self.x(), *self._parameters)

    def covarianceMatrix(self):
        return self._covariance

    def correlationMatrix(self):
        return self._covariance / np.outer(self.uncertainties(), self.uncertainties())
