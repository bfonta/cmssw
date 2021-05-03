#include <iostream>
#include <iomanip>
#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class EcalUncalibRecHitGPUAnalyzer : public DQMEDAnalyzer {
public:
  explicit EcalUncalibRecHitGPUAnalyzer(const edm::ParameterSet &);
  ~EcalUncalibRecHitGPUAnalyzer() override {}

private:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  using InputProductGPU = ecal::UncalibratedRecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using InputProductEECPU = EEUncalibratedRecHitCollection;
  using InputProductEBCPU = EBUncalibratedRecHitCollection;
  edm::EDGetTokenT<InputProductEECPU> cpuEBTok_;
  edm::EDGetTokenT<InputProductEBCPU> cpuEETok_;
  edm::EDGetTokenT<InputProductGPU> gpuEBTok_, gpuEETok_;

  // RecHits plots for EB and EE on both GPU and CPU
  MonitorElement *hRechitsEBGPU = nullptr;
  MonitorElement *hRechitsEBCPU = nullptr;
  MonitorElement *hRechitsEEGPU = nullptr;
  MonitorElement *hRechitsEECPU = nullptr;
  MonitorElement *hRechitsEBGPUCPUratio = nullptr;
  MonitorElement *hRechitsEEGPUCPUratio = nullptr;

  MonitorElement *hSOIAmplitudesEBGPU = nullptr;
  MonitorElement *hSOIAmplitudesEEGPU = nullptr;
  MonitorElement *hSOIAmplitudesEBCPU = nullptr;
  MonitorElement *hSOIAmplitudesEECPU = nullptr;
  MonitorElement *hSOIAmplitudesEBGPUCPUratio = nullptr;
  MonitorElement *hSOIAmplitudesEEGPUCPUratio = nullptr;

  MonitorElement *hChi2EBGPU = nullptr;
  MonitorElement *hChi2EEGPU = nullptr;
  MonitorElement *hChi2EBCPU = nullptr;
  MonitorElement *hChi2EECPU = nullptr;
  MonitorElement *hChi2EBGPUCPUratio = nullptr;
  MonitorElement *hChi2EEGPUCPUratio = nullptr;

  MonitorElement *hFlagsEBGPU = nullptr;
  MonitorElement *hFlagsEEGPU = nullptr;
  MonitorElement *hFlagsEBCPU = nullptr;
  MonitorElement *hFlagsEECPU = nullptr;
  MonitorElement *hFlagsEBGPUCPUratio = nullptr;
  MonitorElement *hFlagsEEGPUCPUratio = nullptr;

  MonitorElement *hSOIAmplitudesEBGPUvsCPU = nullptr;
  MonitorElement *hSOIAmplitudesEEGPUvsCPU = nullptr;
  MonitorElement *hSOIAmplitudesEBdeltavsCPU = nullptr;
  MonitorElement *hSOIAmplitudesEEdeltavsCPU = nullptr;
  
  MonitorElement *hChi2EBGPUvsCPU = nullptr;
  MonitorElement *hChi2EEGPUvsCPU = nullptr;
  MonitorElement *hChi2EBdeltavsCPU = nullptr;
  MonitorElement *hChi2EEdeltavsCPU = nullptr;
  
  MonitorElement *hFlagsEBGPUvsCPU = nullptr;
  MonitorElement *hFlagsEEGPUvsCPU = nullptr;
  MonitorElement *hFlagsEBdeltavsCPU = nullptr;
  MonitorElement *hFlagsEEdeltavsCPU = nullptr;

  MonitorElement *hRechitsEBGPUvsCPU = nullptr;    
  MonitorElement *hRechitsEEGPUvsCPU = nullptr;
  MonitorElement *hRechitsEBdeltavsCPU = nullptr;
  MonitorElement *hRechitsEEdeltavsCPU = nullptr;

  static constexpr int nbins_count = 200;
  static constexpr float last_count = 5000.f;
  static constexpr int nbins_count_delta = 201;

  static constexpr int nbins = 300;
  static constexpr float last = 3000.f;

  static constexpr int nbins_chi2 = 1000;
  static constexpr float last_chi2 = 200.f;

  static constexpr int nbins_flags = 100;
  static constexpr float last_flags = 100.f;
  static constexpr float delta_flags = 20.f;

  static constexpr int nbins_delta = 201;  // use an odd number to center around 0
  static constexpr float delta = 0.2f;

  static constexpr float eps_diff = 1e-3;
};


EcalUncalibRecHitGPUAnalyzer::EcalUncalibRecHitGPUAnalyzer(const edm::ParameterSet &parameters)
  : cpuEBTok_( consumes<InputProductEBCPU>(parameters.getParameter<edm::InputTag>("cpuEBUncalibRecHitCollection")) ),
    cpuEETok_( consumes<InputProductEECPU>(parameters.getParameter<edm::InputTag>("cpuEEUncalibRecHitCollection")) ),
    gpuEBTok_( consumes<InputProductGPU>(parameters.getParameter<edm::InputTag>("gpuEBUncalibRecHitCollection")) ),
    gpuEETok_( consumes<InputProductGPU>(parameters.getParameter<edm::InputTag>("gpuEEUncalibRecHitCollection")) ) {}

// ------------ method called for each event  ------------
void EcalUncalibRecHitGPUAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const auto &hitsEBcpu = iEvent.get(cpuEBTok_);
  const auto &hitsEEcpu = iEvent.get(cpuEETok_);
  const auto &hitsEBgpu = iEvent.get(gpuEBTok_);
  const auto &hitsEEgpu = iEvent.get(gpuEETok_);

  const auto cpu_eb_size = hitsEBcpu.size();
  const auto cpu_ee_size = hitsEEcpu.size();
  auto gpu_eb_size = hitsEBgpu.amplitude.size();
  auto gpu_ee_size = hitsEEgpu.amplitude.size();

  if (cpu_eb_size!=gpu_eb_size or cpu_ee_size!=gpu_ee_size)
    edm::LogError("DifferentSizes_BEFORE_Selection") <<
      " Uncalib EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << gpu_eb_size <<
      " (gpu)\n" <<
      " Uncalib  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << gpu_ee_size <<
      " (gpu)" << std::endl;
  
  float eb_ratio = (float)gpu_eb_size / cpu_eb_size;
  float ee_ratio = (float)gpu_ee_size / cpu_ee_size;
  
  // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
  hRechitsEBGPU->Fill(gpu_eb_size);
  hRechitsEBCPU->Fill(cpu_eb_size);
  hRechitsEEGPU->Fill(gpu_ee_size);
  hRechitsEECPU->Fill(cpu_ee_size);
  hRechitsEBGPUvsCPU->Fill(cpu_eb_size, gpu_eb_size);
  hRechitsEEGPUvsCPU->Fill(cpu_ee_size, gpu_ee_size);
  hRechitsEBGPUCPUratio->Fill(eb_ratio);
  hRechitsEEGPUCPUratio->Fill(ee_ratio);
  hRechitsEBdeltavsCPU->Fill(cpu_eb_size, gpu_eb_size - cpu_eb_size);
  hRechitsEEdeltavsCPU->Fill(cpu_ee_size, gpu_ee_size - cpu_ee_size);

  for (unsigned i=0; i<gpu_eb_size; ++i) {
    auto const did_gpu = hitsEBgpu.did[i];
    auto const soi_amp_gpu = hitsEBgpu.amplitude[i];

    auto const cpu_iter = hitsEBcpu.find(DetId{did_gpu});
    if (cpu_iter == hitsEBcpu.end()) {
      edm::LogError("MissingDetId") <<
	"  Did not find a DetId " << did_gpu << " in a CPU collection\n";
      continue;
    }
    auto const soi_amp_cpu = cpu_iter->amplitude();
    auto const chi2_gpu = hitsEBgpu.chi2[i];
    auto const chi2_cpu = cpu_iter->chi2();

    auto const flags_gpu = hitsEBgpu.flags[i];
    auto const flags_cpu = cpu_iter->flags();

    hSOIAmplitudesEBGPU->Fill(soi_amp_gpu);
    hSOIAmplitudesEBCPU->Fill(soi_amp_cpu);
    hSOIAmplitudesEBGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
    hSOIAmplitudesEBdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
    if (soi_amp_cpu > 0)
      hSOIAmplitudesEBGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);

    hChi2EBGPU->Fill(chi2_gpu);
    hChi2EBCPU->Fill(chi2_cpu);
    hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
    hChi2EBdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);
    if (chi2_cpu > 0)
      hChi2EBGPUCPUratio->Fill((float)chi2_gpu / chi2_cpu);

    if (std::abs(chi2_gpu / chi2_cpu - 1) > 0.05 || std::abs(soi_amp_gpu / soi_amp_cpu - 1) > 0.05) {
      edm::LogError("Mismatch Chi2 or Amplitude") <<
        " ---- EB  " <<
        " xtal = " << i <<
        " chi2_gpu    = " << chi2_gpu << "\n chi2_cpu = " << chi2_cpu <<
        " soi_amp_gpu = " << soi_amp_gpu << "\n soi_amp_cpu = " << soi_amp_cpu <<
        " flags_gpu   = " << flags_gpu << "\n flags_cpu =   " << flags_cpu;
    }

    hFlagsEBGPU->Fill(flags_gpu);
    hFlagsEBCPU->Fill(flags_cpu);
    hFlagsEBGPUvsCPU->Fill(flags_cpu, flags_gpu);
    hFlagsEBdeltavsCPU->Fill(flags_cpu, flags_gpu - flags_cpu);
    if (flags_cpu > 0)
      hFlagsEBGPUCPUratio->Fill((float)flags_gpu / flags_cpu);

    if (flags_cpu != flags_gpu) {
      edm::LogError("Mismatch Flags") <<
        "    >>  No! Different flag cpu:gpu = " << flags_cpu << " : " << flags_gpu;
    }

    if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
	std::isnan(chi2_gpu) or (flags_cpu != flags_gpu)) {
      edm::LogError("LargerThanEps") <<
        "EB chid = " << i << " amp_gpu = " << soi_amp_gpu << " amp_cpu " << soi_amp_cpu <<
	" chi2_gpu = " << chi2_gpu << " chi2_cpu = " << chi2_cpu;
      if (std::isnan(chi2_gpu))
	edm::LogError("NanChi2EB") << "*** nan ***";
    }
  }

  for (unsigned i=0; i<gpu_ee_size; ++i) {
    auto const did_gpu = hitsEEgpu.did[i];
    auto const soi_amp_gpu = hitsEEgpu.amplitude[i];
    auto const cpu_iter = hitsEEcpu.find(DetId{did_gpu});
    if (cpu_iter == hitsEEcpu.end()) {
      edm::LogError("MissingDetId") <<
	"  did not find a DetId " << did_gpu << " in a CPU collection\n";
      continue;
    }
    auto const soi_amp_cpu = cpu_iter->amplitude();
    auto const chi2_gpu = hitsEEgpu.chi2[i];
    auto const chi2_cpu = cpu_iter->chi2();

    auto const flags_gpu = hitsEEgpu.flags[i];
    auto const flags_cpu = cpu_iter->flags();

    hSOIAmplitudesEEGPU->Fill(soi_amp_gpu);
    hSOIAmplitudesEECPU->Fill(soi_amp_cpu);
    hSOIAmplitudesEEGPUvsCPU->Fill(soi_amp_cpu, soi_amp_gpu);
    hSOIAmplitudesEEdeltavsCPU->Fill(soi_amp_cpu, soi_amp_gpu - soi_amp_cpu);
    if (soi_amp_cpu > 0)
      hSOIAmplitudesEEGPUCPUratio->Fill((float)soi_amp_gpu / soi_amp_cpu);

    hChi2EEGPU->Fill(chi2_gpu);
    hChi2EECPU->Fill(chi2_cpu);
    hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
    hChi2EEdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);
    if (chi2_cpu > 0)
      hChi2EEGPUCPUratio->Fill((float)chi2_gpu / chi2_cpu);

    if (std::abs(chi2_gpu / chi2_cpu - 1) > 0.05 || std::abs(soi_amp_gpu / soi_amp_cpu - 1) > 0.05) {
      edm::LogError("Mismatch Chi2 or Amplitude") <<
        " ---- EE  " <<
        " xtal = " << i <<
        " chi2_gpu    = " << chi2_gpu << "\n chi2_cpu =    " << chi2_cpu <<
        " soi_amp_gpu = " << soi_amp_gpu << "\n soi_amp_cpu = " << soi_amp_cpu <<
        " flags_gpu   = " << flags_gpu << "\n flags_cpu =   " << flags_cpu;
    }

    hFlagsEEGPU->Fill(flags_gpu);
    hFlagsEECPU->Fill(flags_cpu);
    hFlagsEEGPUvsCPU->Fill(flags_cpu, flags_gpu);
    hFlagsEEdeltavsCPU->Fill(flags_cpu, flags_gpu - flags_cpu);
    if (flags_cpu > 0)
      hFlagsEEGPUCPUratio->Fill((float)flags_gpu / flags_cpu);

    if (flags_cpu != flags_gpu) {
      edm::LogError("Mismatch flags") <<
        "    >>  No! Different flag cpu:gpu = " << flags_cpu << " : " << flags_gpu;
    }

    if ((std::abs(soi_amp_gpu - soi_amp_cpu) >= eps_diff) or (std::abs(chi2_gpu - chi2_cpu) >= eps_diff) or
	std::isnan(chi2_gpu) or (flags_cpu != flags_gpu)) {
      edm::LogError("LargerThanEps") <<
        "EE chid = " << i << " amp_gpu = " << soi_amp_gpu << " amp_cpu " << soi_amp_cpu <<
	" chi2_gpu = " << chi2_gpu << " chi2_cpu = " << chi2_cpu;
      if (std::isnan(chi2_gpu))
	edm::LogError("NanChi2EE") << "*** nan ***";
    }
  }
}

void EcalUncalibRecHitGPUAnalyzer::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  std::string logTraceName("EcalUncalibRecHitGPUAnalyzer");

  LogTrace(logTraceName) << "Parameters initialization";

  // RecHits plots for EB and EE on both GPU and CPU
  hRechitsEBGPU = iBooker.book1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits", nbins_count, 0, last_count);
  hRechitsEBCPU = iBooker.book1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits", nbins_count, 0, last_count);
  hRechitsEEGPU = iBooker.book1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits", nbins_count, 0, last_count);
  hRechitsEECPU = iBooker.book1D("RechitsEECPU", "RechitsEECPU; No. of Rechits", nbins_count, 0, last_count);
  hRechitsEBGPUCPUratio = iBooker.book1D("RechitsEBGPU_CPUratio", "RechitsEBGPU_CPUratio; GPU/CPU", 50, 0.9, 1.1);
  hRechitsEEGPUCPUratio = iBooker.book1D("RechitsEEGPU_CPUratio", "RechitsEEGPU_CPUratio; GPU/CPU", 50, 0.9, 1.1);

  hSOIAmplitudesEBGPU = iBooker.book1D("hSOIAmplitudesEBGPU", "hSOIAmplitudesEBGPU", nbins, 0, last);
  hSOIAmplitudesEEGPU = iBooker.book1D("hSOIAmplitudesEEGPU", "hSOIAmplitudesEEGPU", nbins, 0, last);
  hSOIAmplitudesEBCPU = iBooker.book1D("hSOIAmplitudesEBCPU", "hSOIAmplitudesEBCPU", nbins, 0, last);
  hSOIAmplitudesEECPU = iBooker.book1D("hSOIAmplitudesEECPU", "hSOIAmplitudesEECPU", nbins, 0, last);
  hSOIAmplitudesEBGPUCPUratio = iBooker.book1D("SOIAmplitudesEBGPU_CPUratio", "SOIAmplitudesEBGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);
  hSOIAmplitudesEEGPUCPUratio = iBooker.book1D("SOIAmplitudesEEGPU_CPUratio", "SOIAmplitudesEEGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);

  hChi2EBGPU = iBooker.book1D("hChi2EBGPU", "hChi2EBGPU", nbins_chi2, 0, last_chi2);
  hChi2EEGPU = iBooker.book1D("hChi2EEGPU", "hChi2EEGPU", nbins_chi2, 0, last_chi2);
  hChi2EBCPU = iBooker.book1D("hChi2EBCPU", "hChi2EBCPU", nbins_chi2, 0, last_chi2);
  hChi2EECPU = iBooker.book1D("hChi2EECPU", "hChi2EECPU", nbins_chi2, 0, last_chi2);
  hChi2EBGPUCPUratio = iBooker.book1D("Chi2EBGPU_CPUratio", "Chi2EBGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);
  hChi2EEGPUCPUratio = iBooker.book1D("Chi2EEGPU_CPUratio", "Chi2EEGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);

  hFlagsEBGPU = iBooker.book1D("hFlagsEBGPU", "hFlagsEBGPU", nbins_flags, 0, last_flags);
  hFlagsEEGPU = iBooker.book1D("hFlagsEEGPU", "hFlagsEEGPU", nbins_flags, 0, last_flags);
  hFlagsEBCPU = iBooker.book1D("hFlagsEBCPU", "hFlagsEBCPU", nbins_flags, 0, last_flags);
  hFlagsEECPU = iBooker.book1D("hFlagsEECPU", "hFlagsEECPU", nbins_flags, 0, last_flags);
  hFlagsEBGPUCPUratio = iBooker.book1D("FlagsEBGPU_CPUratio", "FlagsEBGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);
  hFlagsEEGPUCPUratio = iBooker.book1D("FlagsEEGPU_CPUratio", "FlagsEEGPU_CPUratio; GPU/CPU", 200, 0.9, 1.1);
  
  hSOIAmplitudesEBGPUvsCPU = iBooker.book2D("hSOIAmplitudesEBGPUvsCPU", "hSOIAmplitudesEBGPUvsCPU", nbins, 0, last, nbins, 0, last);
  hSOIAmplitudesEEGPUvsCPU = iBooker.book2D("hSOIAmplitudesEEGPUvsCPU", "hSOIAmplitudesEEGPUvsCPU", nbins, 0, last, nbins, 0, last);
  hSOIAmplitudesEBdeltavsCPU = iBooker.book2D("hSOIAmplitudesEBdeltavsCPU", "hSOIAmplitudesEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  hSOIAmplitudesEEdeltavsCPU = iBooker.book2D("hSOIAmplitudesEEdeltavsCPU", "hSOIAmplitudesEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  hChi2EBGPUvsCPU = iBooker.book2D("hChi2EBGPUvsCPU", "hChi2EBGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  hChi2EEGPUvsCPU = iBooker.book2D("hChi2EEGPUvsCPU", "hChi2EEGPUvsCPU", nbins_chi2, 0, last_chi2, nbins_chi2, 0, last_chi2);
  hChi2EBdeltavsCPU = iBooker.book2D("hChi2EBdeltavsCPU", "hChi2EBdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  hChi2EEdeltavsCPU = iBooker.book2D("hChi2EEdeltavsCPU", "hChi2EEdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);

  hFlagsEBGPUvsCPU = iBooker.book2D("hFlagsEBGPUvsCPU", "hFlagsEBGPUvsCPU", nbins_flags, 0, last_flags, nbins_flags, 0, last_flags);
  hFlagsEEGPUvsCPU = iBooker.book2D("hFlagsEEGPUvsCPU", "hFlagsEEGPUvsCPU", nbins_flags, 0, last_flags, nbins_flags, 0, last_flags);
  hFlagsEBdeltavsCPU = iBooker.book2D( "hFlagsEBdeltavsCPU", "hFlagsEBdeltavsCPU", nbins_flags, 0, last_flags, nbins_delta, -delta_flags, delta_flags);
  hFlagsEEdeltavsCPU = iBooker.book2D("hFlagsEEdeltavsCPU", "hFlagsEEdeltavsCPU", nbins_flags, 0, last_flags, nbins_delta, -delta_flags, delta_flags);
  
  hRechitsEBGPUvsCPU = iBooker.book2D("RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU", last_count, 0, last_count, last_count, 0, last_count);
  hRechitsEEGPUvsCPU = iBooker.book2D("RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU", last_count, 0, last_count, last_count, 0, last_count);
  hRechitsEBdeltavsCPU = iBooker.book2D("RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU", nbins_count, 0, last_count, nbins_count_delta, -delta, delta);
  hRechitsEEdeltavsCPU = iBooker.book2D("RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU", nbins_count, 0, last_count, nbins_count_delta, -delta, delta);
}

DEFINE_FWK_MODULE(EcalUncalibRecHitGPUAnalyzer);
