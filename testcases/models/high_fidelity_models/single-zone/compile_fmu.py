# -*- coding: utf-8 -*-
"""
This module compiles Modelica models into fmu models.

@ Yangyang Fu, yangyang.fu@tamu.edu

"""


from pymodelica import compile_fmu
# DEFINE MODEL
# ------------
mopath = 'SingleZoneVAV.mo'
modelpath = 'SingleZoneVAV.AirflowBaseline'
# ------------

# COMPILE FMU: set JVM maximum leap to 1G to avoid memory issues
# -----------
# the defauted compilation target is model exchange, where no numerical integrator is integrated into the fmu. 
# The equations in FMU is solved by numerical solvers in the importing tool.
compiler_options = {"cs_rel_tol":1.0E-04}
fmupath = compile_fmu(modelpath,[mopath], jvm_args='-Xmx1g',target='cs',version='2.0',compile_to='SingleZoneDamperControlBaseline.fmu',compiler_options=compiler_options)
# -----------



