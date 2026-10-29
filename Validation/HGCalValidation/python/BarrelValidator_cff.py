import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
from Validation.HGCalValidation.barrelValidator_cfi import barrelValidator as _barrelValidator
from RecoTICL.Configuration.iterativeTICL_cff import ticlBarrelIterLabelsPSet

# The associations of each barrel trackster collection with the barrel sim tracksters, both ways
_simTrackstersBarrel = ['ticlSimTrackstersBarrel', 'ticlSimTrackstersBarrelfromCPs']
def _associators(producer):
    return [cms.InputTag(producer, instance)
            for label in ticlBarrelIterLabelsPSet.labels
            for sim in _simTrackstersBarrel
            for instance in (f'{label}To{sim}', f'{sim}To{label}')]

barrelValidator = _barrelValidator.clone(
    label_tst = [cms.InputTag(label) for label in ticlBarrelIterLabelsPSet.labels],
    allTracksterTracksterAssociatorsLabels = _associators('allBarrelTrackstersToSimTrackstersAssociationsByLCs'),
    allTracksterTracksterByHitsAssociatorsLabels = _associators('allBarrelTrackstersToSimTrackstersAssociationsByHits'),
)
