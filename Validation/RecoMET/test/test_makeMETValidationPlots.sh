#!/bin/bash

# Usage: test_makeMETValidationPlots.sh [workflow id] (or WORKFLOW=<id> test_makeMETValidationPlots.sh)

function die { echo "$1: status $2"; exit $2; }

LOCAL_TEST_DIR="${LOCAL_TEST_DIR:-${PWD}}"
NEVENTS=10
NTHREADS=4
SCRIPT="makeMETValidationPlots.py"

DEFAULT_WORKFLOW="38734.101" # RelValWprimeToLNu_M2000_14TeV
WORKFLOW="${1:-${WORKFLOW:-${DEFAULT_WORKFLOW}}}"

# HLT part: not part of the offline matrix workflow, hence pinned here
HLTSEQ="75e33"
HLTVALIDSEQ="@hltValidation"
HLTPROC="HLTX"

REDIRECTOR="root://eoscms.cern.ch/" # root://cms-xrd-global.cern.ch
INPUT="/store/relval/CMSSW_20_0_0_patch1/RelValWprimeToLNu_M2000_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D128_RegeneratedGS_PU_16Aug26-v2/2590000/feacc96a-1105-475a-8c26-4ea4e0d00354.root"

xrdfs ${REDIRECTOR} locate ${INPUT} > /dev/null 2>&1
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    die "SKIPPING test, input ${INPUT} not reachable through ${REDIRECTOR}" 0
fi

WORKDIR=$(mktemp -d "${LOCAL_TEST_DIR}/metvalidation.XXXXXX") || die "cannot create work directory" 1
# the work directory (with the logs of each step) is only removed if the test succeeds
trap 'rc=$?; if [ ${rc} -eq 0 ]; then rm -rf "${WORKDIR}"; else echo "Test failed, keeping ${WORKDIR}"; fi' EXIT
cd "${WORKDIR}" || die "cannot enter ${WORKDIR}" 1
printf "Using workflow ${WORKFLOW}, input ${INPUT}.\nRunning in ${WORKDIR}.\n"

# ---------------------------------------------------------------------------------------------------------
# Parse the cmsDriver.py commands of the workflow
# ---------------------------------------------------------------------------------------------------------
mkdir matrix && (cd matrix && runTheMatrix.py -w upgrade -l ${WORKFLOW} --dryRun > matrix.log 2>&1) \
    || die "runTheMatrix.py failed for workflow ${WORKFLOW}" $?
RECOCMD=$(grep -h "cmsDriver.py step3" matrix/matrix.log | head -n 1)
[ -n "${RECOCMD}" ] || die "workflow ${WORKFLOW} has no RECO (step3) cmsDriver.py command" 1

# value of option $2 in command $1
function opt { echo "$1" | sed -nE "s/.* $2 +([^ ]+).*/\1/p"; }

GEOM=$(opt "${RECOCMD}" --geometry)
ERA=$(opt "${RECOCMD}" --era)
CONDS=$(opt "${RECOCMD}" --conditions)
AGING=$(opt "${RECOCMD}" --customise)
RECOSEQ=$(opt "${RECOCMD}" -s)       # RAW2DIGI,RECO,...,VALIDATION:@phase2Validation+...,DQM:...
# MET harvesting only: the harvesting of the full @phase2Validation expects the histograms of all validations
HARVSEQ="HARVESTING:METPostProcessor"
# Only validate MET: the full @phase2Validation also needs inputs this sample does not have (e.g. the truth graph)
# (cmsDriver.py only accepts a list of sequences joined by '+' if they are '@' aliases)
RECOSEQ=$(echo "${RECOSEQ}" | sed -E 's/VALIDATION:[^,]*/VALIDATION:METValidation/')
for v in GEOM ERA CONDS AGING RECOSEQ HARVSEQ; do
    [ -n "${!v}" ] || die "could not parse ${v} from workflow ${WORKFLOW}" 1
done

# MHT cut scan (see addMHTScanHLT/addMHTScanOffline), with the per-pdgId folders of W' -> l nu
HLTSCAN="from Validation.RecoMET.hltMETValidation_cff import addMHTScanHLT; addMHTScanHLT(process, genWDecayPdgIds=(11,13,15))"
OFFSCAN="from Validation.RecoMET.METValidation_cfi import addMHTScanOffline; addMHTScanOffline(process, genWDecayPdgIds=(11,13,15))"

# ---------------------------------------------------------------------------------------------------------
# HLT: step2 (HLT + validation) and harvesting
# ---------------------------------------------------------------------------------------------------------
(cmsDriver.py step2 -s L1P2GT,HLT:"${HLTSEQ}",DQM:trackingMonitorHLT,VALIDATION:"${HLTVALIDSEQ}" \
              --conditions "${CONDS}" \
              --datatier GEN-SIM-DIGI-RAW,DQMIO \
              -n ${NEVENTS} \
              --eventcontent FEVTDEBUGHLT,DQMIO \
              --geometry "${GEOM}" \
              --era "${ERA}" \
              --customise "${AGING}" \
              --customise_commands "${HLTSCAN}" \
              --filein ${REDIRECTOR}/${INPUT} \
              --fileout file:hlt_step2.root \
              --nThreads ${NTHREADS} \
              --hltProcess ${HLTPROC} \
              --processName ${HLTPROC} \
              --inputCommands='keep *, drop *_hlt*_*_HLT, keep *_hltParticleFlowTmpBarrel_*_*, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
              --outputCommands='keep *_hltParticleFlowTmpBarrel_*_HLTX, keep *_hltParticleFlowTmp_*_HLTX' \
              --python_filename hlt_step2_cfg.py > hlt_step2.log 2>&1) || die "HLT step2 failed (see hlt_step2.log)" $?

(cmsDriver.py step3 -s HARVESTING:"${HLTVALIDSEQ}" \
              --conditions "${CONDS}" \
              --mc \
              --geometry "${GEOM}" \
              --scenario pp \
              --filetype DQM \
              --era "${ERA}" \
              -n ${NEVENTS} \
              --hltProcess ${HLTPROC} \
              --filein file:hlt_step2_inDQM.root \
              --fileout file:hlt_step3.root \
              --python_filename hlt_step3_cfg.py > hlt_step3.log 2>&1) || die "HLT harvesting failed (see hlt_step3.log)" $?
# the harvesting names its output DQM_V<version>_R<run>__<dataset>.root: rename it, as the offline one is produced next
HLTDQM="hlt_dqm.root"
mv DQM_V*.root ${HLTDQM} || die "no HLT DQM file produced (see hlt_step3.log)" 1

# ---------------------------------------------------------------------------------------------------------
# Offline: RECO + validation and harvesting (the input is already pileup-mixed, hence no --pileup)
# ---------------------------------------------------------------------------------------------------------
(cmsDriver.py step3 -s "${RECOSEQ}" \
              --conditions "${CONDS}" \
              --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO \
              -n ${NEVENTS} \
              --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM \
              --geometry "${GEOM}" \
              --era "${ERA}" \
              --customise "${AGING}" \
              --customise_commands "${OFFSCAN}" \
              --filein ${REDIRECTOR}/${INPUT} \
              --fileout file:offline_step3.root \
              --nThreads ${NTHREADS} \
              --python_filename offline_step3_cfg.py > offline_step3.log 2>&1) || die "offline RECO failed (see offline_step3.log)" $?

(cmsDriver.py step4 -s "${HARVSEQ}" \
              --conditions "${CONDS}" \
              --mc \
              --geometry "${GEOM}" \
              --scenario pp \
              --filetype DQM \
              --era "${ERA}" \
              -n ${NEVENTS} \
              --filein file:offline_step3_inDQM.root \
              --fileout file:offline_step4.root \
              --python_filename offline_step4_cfg.py > offline_step4.log 2>&1) || die "offline harvesting failed (see offline_step4.log)" $?
OFFDQM="offline_dqm.root"
mv DQM_V*.root ${OFFDQM} || die "no offline DQM file produced (see offline_step4.log)" 1

# ---------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------
# collections booked in each DQM file (folders of the METValidation directory), as read from the file itself
function collections { python3 -I - "$1" "$2" <<'EOF'
import sys, ROOT
f = ROOT.TFile.Open(sys.argv[1])
d = f.Get(sys.argv[2])
print(' '.join(k.GetName() for k in d.GetListOfKeys() if k.IsFolder() and k.GetName() not in ('METvsMHT', 'genMetTrue')))
EOF
}
HLTCOLLS=$(collections "${HLTDQM}" "DQMData/Run 1/HLT/Run summary/JetMET/METValidation")
OFFCOLLS=$(collections "${OFFDQM}" "DQMData/Run 1/JetMET/Run summary/METValidation")
echo "HLT collections: ${HLTCOLLS}"
echo "Offline collections: ${OFFCOLLS}"

${SCRIPT} --file ./${HLTDQM} --odir ./hlt --step HLT --sample_label Wprime --overlay_scan --met ${HLTCOLLS} \
    || die "failed running ${SCRIPT} on the HLT file" $?
${SCRIPT} --file ./${OFFDQM} --odir ./offline --step Offline --sample_label Wprime --overlay_scan --met ${OFFCOLLS} \
    || die "failed running ${SCRIPT} on the offline file" $?
${SCRIPT} --file ./${HLTDQM} --offline_file ./${OFFDQM} --odir ./comparison --sample_label Wprime \
    --compare_hlt_offline hltPFPuppiMET:pfMetPuppi hltPFPuppiMHT:pfPuppiMHT hltPFMET:pfMet \
    || die "failed running ${SCRIPT} to compare the HLT and offline files" $?
