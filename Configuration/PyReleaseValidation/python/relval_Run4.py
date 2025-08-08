# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done.
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

#just define all of them
prefixDet = 29600 #update this line when change the default version

#Run4 WFs to run in IB (TTbar)
numWFIB = [
    23634.0, #Run4D95
    24034.0, #Run4D96
    24834.0, #Run4D98
    25234.0, #Run4D99
    25634.0, #Run4D100
    26034.0, #Run4D101
    26434.0, #Run4D102
    26834.0, #Run4D103
    27234.0, #Run4D104
    27634.0, #Run4D105
    28034.0, #Run4D106
    28434.0, #Run4D107
    28834.0, #Run4D108
    29234.0, #Run4D109
    29634.0, #Run4D110
    30034.0, #Run4D111
    30434.0, #Run4D112
    30834.0, #Run4D113
    31234.0, #Run4D114
    32034.0, #Run4D115
    32434.0, #Run4D116
    32834.0, #Run4D117
    33234.0, #Run4D118
    33634.0, #Run4D119
    34034.0, #Run4D120
    34434.0, #Run4D121

    #Additional sample for short matrix and IB
    #Default Phase-2 Det NoPU
    prefixDet+34.911, #DD4hep XML
    prefixDet+34.702, #mkFit tracking (initialStep)
    prefixDet+34.5,   #pixelTrackingOnly
    prefixDet+34.9,   #vector hits
    prefixDet+34.402, #Alpaka local reconstruction offloaded on device (GPU if available)
    prefixDet+34.703, #LST tracking on CPU (initialStep+HighPtTripletStep only)
    prefixDet+34.21,  #prodlike
    prefixDet+96.0,   #CloseByPGun CE_E_Front_120um
    prefixDet+100.0,  #CloseByPGun CE_H_Coarse_Scint
    prefixDet+61.0,   #Nu Gun
    prefixDet+34.75,  #Timing menu
    prefixDet+151.85, #Heavy ion reconstruction
    #Default Phase-2 Det PU
    prefixDet+261.97,   #premixing stage1 (NuGun+PU)
    prefixDet+234.99,   #premixing combined stage1+stage2 ttbar+PU200
    prefixDet+234.999,  #premixing combined stage1+stage2 ttbar+PU50 for PR test
    prefixDet+234.21,   #prodlike PU
    prefixDet+234.9921, #prodlike premix stage1+stage2
    prefixDet+234.114,  #PU, with 10% OT inefficiency
    prefixDet+234.703,  #LST tracking on CPU (initialStep+HighPtTripletStep only)
    #
    24834.911, #D98 XML, to monitor instability of DD4hep
    
    # Phase-2 HLT tests
    prefixDet+34.751, # HLTTiming75e33, alpaka
    prefixDet+34.752, # HLTTiming75e33, ticl_v5
    prefixDet+34.753, # HLTTiming75e33, alpaka,singleIterPatatrack
    prefixDet+34.754, # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST
    prefixDet+34.755, # HLTTiming75e33, alpaka,trackingLST
    prefixDet+34.756, # HLTTiming75e33, phase2_hlt_vertexTrimming
    prefixDet+34.7561,# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming
    prefixDet+34.7562,# HLTTiming75e33, alpaka,phase2_hlt_vertexTrimming,singleIterPatatrack
    prefixDet+34.757, # HLTTiming75e33, alpaka,singleIterPatatrack,trackingLST,seedingLST
    prefixDet+34.758, # HLTTiming75e33, ticl_barrel
    prefixDet+34.759, # HLTTiming75e33 + NANO
    prefixDet+34.77,  # NGTScouting
    prefixDet+34.771, # NGTScouting + alpaka + TICL-v5 + TICL-Barrel
    prefixDet+34.772, # NGTScouting + NANO
    prefixDet+34.773, # NGTScouting + NANO (including validation)
]

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
