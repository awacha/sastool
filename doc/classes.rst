SAS Classes
===========

`sastool` implements a versatile, high-level interface to one- and two-dimensional
scattering data in several Python classes:

   `SASHeader`
      represents the meta-data of a scattering experiment, such as wavelength,
      particle type, detector dimensions, sample-to-detector distance, sample
      name etc.
      
   `SASExposure`
      represents the two-dimensional scattering dataset. Usually accompanied by
      a `SASHeader` and a `SASMask` instance, this enables several powerful
      operations to be carried out, such as beam position finding, advanced
      plotting, radial and azimuthal averaging, input from various formats,
      basic arithmetics etc.
   
   `SASMask`
      the Python representation of a mask matrix. Basic input/output and various
      editing functions (circle, polygon, rectangle etc.) are supported.
      
   `GeneralCurve` and its subclasses
      these represent a one-dimensional dataset, e.g. a scattering curve. Basic
      input-output (from and to text files) and conversions (e.g. from Python
      `dict`-s) are supported. Matplotlib-style plotting is ensured via the
      corresponding methods. An advanced fitting engine helps the interpretation
      of the scattering results.
      

.. toctree::

   header
   exposure
   mask
   curve

