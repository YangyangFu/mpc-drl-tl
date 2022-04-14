# -*- coding: utf-8 -*-
"""
This module compiles Modelica models into fmu models.

@ Yangyang Fu, yangyang.fu@tamu.edu

"""
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from pymodelica import compile_fmu

#### Compile Baseline Model
## ------------
mopath = 'SingleZoneFanCoilUnit.mo'
modelpath = 'SingleZoneFanCoilUnit.TestCases.Baseline'
# ------------

# COMPILE FMU: set JVM maximum leap to 1G to avoid memory issues
# -----------
# the defauted compilation target is model exchange, where no numerical integrator is integrated into the fmu. 
# The equations in FMU is solved by numerical solvers in the importing tool.
compiler_options = {"cs_rel_tol":1.0E-04}
fmupath = compile_fmu(modelpath,[mopath], jvm_args='-Xmx1g',target='cs',version='2.0',compile_to='SingleZoneFCUBaseline.fmu',compiler_options=compiler_options)
# -----------

#### Compile AirFlow
# ---------------------
mopath = 'SingleZoneFanCoilUnit.mo'
modelpath = 'SingleZoneFanCoilUnit.TestCases.FanControl'
# ------------

# COMPILE FMU: set JVM maximum leap to 1G to avoid memory issues
# -----------
# the defauted compilation target is model exchange, where no numerical integrator is integrated into the fmu. 
# The equations in FMU is solved by numerical solvers in the importing tool.
fmupath = compile_fmu(modelpath,[mopath], jvm_args='-Xmx1g',target='cs',version='2.0',compile_to='SingleZoneFCU.fmu',compiler_options=compiler_options)
# -----------



