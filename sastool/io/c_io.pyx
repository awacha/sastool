'CPU-intensive or speed-critical parts for io.py written in Cython '
# pylint: disable-msg-cat=WCREFI
cimport numpy as np
import numpy as np

def cbfdecompress(datain, Py_ssize_t dim1, Py_ssize_t dim2, bint for_nexus=False):
    cdef Py_ssize_t index_input
    cdef Py_ssize_t index_output
    cdef long value_current
    cdef long value_diff
    cdef Py_ssize_t nbytes
    cdef np.ndarray[ndim=2,dtype=np.double_t] outarray
    cdef Py_ssize_t npixels
    cdef bytearray data
    
    data=datain
    index_input=0
    index_output=0
    value_current=0
    value_diff=0
    nbytes=len(data)
    npixels=dim1*dim2
    outarray=np.zeros((dim2,dim1),dtype=np.double)
    while(index_input < nbytes):
        value_diff=data[index_input]
        index_input+=1
        if value_diff !=0x80:
            if value_diff>=0x80:
                value_diff=value_diff -0x100
        else: 
            if not ((data[index_input]==0x00 ) and 
                (data[index_input+1]==0x80)):
                value_diff=data[index_input]+\
                            0x100*data[index_input+1]
                if value_diff >=0x8000:
                    value_diff=value_diff-0x10000
                index_input+=2
            else:
                index_input+=2
                value_diff=data[index_input]+\
                           0x100*data[index_input+1]+\
                           0x10000*data[index_input+2]+\
                           0x1000000*data[index_input+3]
                if value_diff>=0x80000000L:
                    value_diff=value_diff-4294967296
                index_input+=4
        value_current+=value_diff
        if index_output<dim1*dim2:
            # if we start the counting of the columns and rows from top left, then
            #     column_nr = index_output % dim1
            #     row_nr = index_output // dim1
            #
            # The index of the output array is column_nr+row_nr*dim1
            if not for_nexus:
                outarray[index_output//dim1, index_output%dim1]=value_current
            else:
                # for NeXus, reverse the row number
                outarray[(dim2-1 - index_output // dim1), index_output % dim1 ]=value_current
        else:
            print "End of output array. Remaining input bytes:", len(data)-index_input
            print "remaining buffer:",data[index_input:]
            break
        index_output+=1
    if index_output != dim1*dim2:
        raise ValueError, "Binary data does not have enough points."
    return outarray

