# -*- coding: utf-8 -*-
"""
This module compiles the defined test case model into an FMU.

"""
from __future__ import print_function
from __future__ import absolute_import, division

from pymodelica import compile_fmu
# DEFINE MODEL
# ------------
library = 'FiveZoneVAV'
modelpath = 'FiveZoneVAV.Guideline36TSup'
# ------------

# COMPILE FMU: set JVM maximum leap to 5G to avoid memory issues
# -----------
fmupath = compile_fmu(modelpath, library, jvm_args='-Xmx3g',target='cs',version='2.0',compile_to='VAV.fmu')
# -----------



