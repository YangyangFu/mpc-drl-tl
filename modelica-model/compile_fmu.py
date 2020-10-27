# -*- coding: utf-8 -*-
"""
This module compiles the defined test case model into an FMU.

"""

from pymodelica import compile_fmu
# DEFINE MODEL
# ------------
library = 'FiveVAV'
modelpath = 'FiveVAV.VAVMPC'
# ------------

# COMPILE FMU: set JVM maximum leap to 5G to avoid memory issues
# -----------
fmupath = compile_fmu(modelpath, library, jvm_args='-Xmx5g',compile_to='VAV.fmu')
# -----------



