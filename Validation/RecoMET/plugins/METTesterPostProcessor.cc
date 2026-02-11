#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"

METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet &iConfig) {
  runDir = iConfig.getUntrackedParameter<std::string>("runDir");
}
METTesterPostProcessor::~METTesterPostProcessor() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessor::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  std::vector<std::string> subDirVec;
  std::string RunDir = runDir;
  iget_.setCurrentFolder(RunDir);
  met_dirs = iget_.getSubdirs();

  auto getData = [](const auto &arr) { return arr.data(); };
  
  // loop over met subdirectories
  for (size_t i = 0; i < met_dirs.size(); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);
    for (std::string bt : {"MET", "Phi"}) {  // loop over bin types
	  // differences
      mMETDiffAggr[bt] = ibook_.book1D("METDiffAggr_" + bt,
                                       "METDiffAggr_" + bt,
                                       mNBins[bt], std::visit(getData, mEdges[bt]));
      mMETDeltaPhiAggr[bt] = ibook_.book1D("METDeltaPhiAggr_" + bt,
										   "METDeltaPhiAggr_" + bt,
										   mNBins[bt], std::visit(getData, mEdges[bt]));

	  // response
      mMETRespAggr[bt] = ibook_.book1D("METRespAggr_" + bt,
                                       "METRespAggr_" + bt,
                                       mNBins[bt], std::visit(getData, mEdges[bt]));
	}
	
	// resolution
	mMETResolAggr["Phi"] = ibook_.book1D("METResolAggr_Phi",
										 "METResolAggr_Phi",
										 mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	mMETGenResolAggr["Phi"] = ibook_.book1D("METGenResolAggr_Phi",
											"METGenResolAggr_Phi",
											mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	mMETResolDiffAggr["Phi"] = ibook_.book1D("METResolDiffAggr_Phi",
											 "METResolDiffAggr_Phi",
											 mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	
	// significance
	mMETSignAggr["Phi"] = ibook_.book1D("METSignAggr_Phi",
										"METSignAggr_Phi",
										mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	mMETGenSignAggr["Phi"] = ibook_.book1D("METGenSignAggr_Phi",
										   "METGenSignAggr_Phi",
										   mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	mMETSignDiffAggr["Phi"] = ibook_.book1D("METSignDiffAggr_Phi",
											"METSignDiffAggr_Phi",
											mNBins["Phi"], std::visit(getData, mEdges["Phi"]));
	
	mFillAggrHistograms(met_dirs[i], iget_);
  }
}

bool METTesterPostProcessor::mCheckHisto(MElem *h) { return h && h->getRootObject(); }

void METTesterPostProcessor::mFillAggrHistograms(std::string metdir, DQMStore::IGetter &iget) {
  for (std::string bt : {"MET", "Phi"}) {  // loop over bin types
    for (unsigned idx = 0; idx < mNBins[bt]; ++idx) {
      std::string edges =
          METTester::binStr(mArrayIdx<float>(mEdges[bt], idx), mArrayIdx<float>(mEdges[bt], idx + 1), bt == "MET");
	  mArrayIdx<MElem *>(mGenMETTrue[bt], idx) = iget.get(metdir + "/GenMETTrue_" + bt + edges);
      mArrayIdx<MElem *>(mMET[bt], idx) = iget.get(metdir + "/MET_" + bt + edges);
      mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx) = iget.get(metdir + "/METDiff_GenMETTrue_" + bt + edges);
      mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx) = iget.get(metdir + "/METRatio_GenMETTrue_" + bt + edges);
      mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx) = iget.get(metdir + "/METDeltaPhi_GenMETTrue_" + bt + edges);

      // check one object, if it exists, then the remaining ME's exists too
      // for genmet none of these ME's are filled
      if (mCheckHisto(mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], 0))) {
        // log histograms with zero entries
        if (mArrayIdx<MElem *>(mGenMETTrue[bt], idx)->getEntries() < mEpsilonDouble ||
			mArrayIdx<MElem *>(mMET[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble ||
            mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getEntries() < mEpsilonDouble) {
          LogDebug("METTesterPostProcessor")
              << "At least one of the " << bt + edges << " histograms has zero entries:\n"
			  << "  Gen MET: " << mArrayIdx<MElem *>(mGenMETTrue[bt], idx)->getEntries() << "\n"
              << "  MET: " << mArrayIdx<MElem *>(mMET[bt], idx)->getEntries() << "\n"
              << "  METDiff: " << mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getEntries() << "\n"
              << "  METRatio: " << mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getEntries() << "\n"
              << "  METDeltaPhi: " << mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getEntries();
        }
      }
    }

    if (mCheckHisto(mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], 0))) {
      // compute and store MET quantities
      for (unsigned idx = 0; idx < mNBins[bt]; ++idx) {
		// difference between reconstructed and generated MET
        mMETDiffAggr[bt]->setBinContent(idx + 1, mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getMean());
        mMETDiffAggr[bt]->setBinError(idx + 1, mArrayIdx<MElem *>(mMETDiff_GenMETTrue[bt], idx)->getRMS());

		mMETDeltaPhiAggr[bt]->setBinContent(idx + 1, mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getMean());
        mMETDeltaPhiAggr[bt]->setBinError(idx + 1, mArrayIdx<MElem *>(mMETDeltaPhi_GenMETTrue[bt], idx)->getRMS());

		// ratio between reconstructed and generated MET
        float ratioMean = mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getMean();
        float ratioRMS = mArrayIdx<MElem *>(mMETRatio_GenMETTrue[bt], idx)->getRMS();
        mMETRespAggr[bt]->setBinContent(idx + 1, ratioMean);
        mMETRespAggr[bt]->setBinError(idx + 1, ratioRMS);

		if (bt == "Phi") {
		  // reconstructed MET
		  float metMean = mArrayIdx<MElem *>(mMET[bt], idx)->getMean();
		  float metRMS = mArrayIdx<MElem *>(mMET[bt], idx)->getRMS();
		  float resolError = mArrayIdx<MElem *>(mMET[bt], idx)->getRMSError();
		  mMETResolAggr[bt]->setBinContent(idx + 1, metRMS);
		  mMETResolAggr[bt]->setBinError(idx + 1, resolError);

		  float significance = metRMS < mEpsilonFloat ? 0.f : metMean / metRMS;
		  float significanceError = (metRMS < mEpsilonFloat || metMean < mEpsilonFloat) ? 0.f : mComputeSignErr(significance, metRMS, metMean, resolError);
		  mMETSignAggr[bt]->setBinContent(idx + 1, significance);
		  mMETSignAggr[bt]->setBinError(idx + 1, significanceError);

		  // generated MET
		  float metGenMean = mArrayIdx<MElem *>(mGenMETTrue[bt], idx)->getMean();
		  float metGenRMS = mArrayIdx<MElem *>(mGenMETTrue[bt], idx)->getRMS();
		  float resolGenError = mArrayIdx<MElem *>(mGenMETTrue[bt], idx)->getRMSError();
		  mMETGenResolAggr[bt]->setBinContent(idx + 1, metGenRMS);
		  mMETGenResolAggr[bt]->setBinError(idx + 1, resolGenError);

		  float significanceGen = metGenRMS < mEpsilonFloat ? 0.f : metGenMean / metGenRMS;
		  float significanceGenError = (metGenRMS < mEpsilonFloat || metGenMean < mEpsilonFloat) ? 0.f : mComputeSignErr(significanceGen, metGenRMS, metGenMean, resolGenError);
		  mMETGenSignAggr[bt]->setBinContent(idx + 1, significanceGen);
		  mMETGenSignAggr[bt]->setBinError(idx + 1, significanceGenError);

		  // comparison between reconstructed and generated MET
		  mMETResolDiffAggr[bt]->setBinContent(idx + 1, metRMS - metGenRMS);
		  mMETResolDiffAggr[bt]->setBinError(idx + 1, std::sqrt(resolGenError*resolGenError + resolError*resolError));

		  mMETSignDiffAggr[bt]->setBinContent(idx + 1, significance - significanceGen);
		  mMETSignDiffAggr[bt]->setBinError(idx + 1, std::sqrt(significanceError*significanceError + significanceGenError*significanceGenError));
		}
      }
    }
  }
}

// Compute significance error
float METTesterPostProcessor::mComputeSignErr(float significance, float metRMS, float metMean, float metRMSError) {
  return significance * std::sqrt((metRMS * metRMS / (metMean * metMean)) +
								  (metRMSError * metRMSError / (metRMS * metRMS)));
}

void METTesterPostProcessor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("runDir", "JetMET/METValidation");
  descriptions.addWithDefaultLabel(desc);
}
