# -*- coding: utf-8 -*-

## as of now only test imports
## testing of GUI to be done after installation manually with example case
# import pytest
# if __name__ == '__main__':     #enclosing required because of multiprocessing

import numpy
print("LaueTools passed import")

def test_method1():
    ##DUmmy test 
    	a = 6
    	b = 8
    	assert a+2== b, "test failed"
    	assert b-2 == a, "test failed"
    
    #TODO add an example test case that verifies all the functionality of GUI
    # For eample run the automated scripts from the example notebook directory
    # need to add prediction routine , with simulated data ?
