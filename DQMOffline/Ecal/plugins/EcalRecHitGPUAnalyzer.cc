#include <iostream>
#include <memory>

// user include files
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"

class EcalRecHitGPUAnalyzer : public DQMEDAnalyzer {
public:
  explicit EcalRecHitGPUAnalyzer(const edm::ParameterSet &);
  ~EcalRecHitGPUAnalyzer() override {}

private:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  using InputProductGPU = ecal::RecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  using InputProductEECPU = EERecHitCollection;
  using InputProductEBCPU = EBRecHitCollection;
  edm::EDGetTokenT<InputProductEECPU> cpuEBTok_;
  edm::EDGetTokenT<InputProductEBCPU> cpuEETok_;
  edm::EDGetTokenT<InputProductGPU> gpuEBTok_, gpuEETok_;

  // RecHits plots for EB and EE on both GPU and CPU
  MonitorElement *hRechitsEBGPU = nullptr;
  MonitorElement *hRechitsEBCPU = nullptr;
  MonitorElement *hRechitsEEGPU = nullptr;
  MonitorElement *hRechitsEECPU = nullptr;
  MonitorElement *hRechitsEBGPUvsCPU = nullptr;
  MonitorElement *hRechitsEEGPUvsCPU = nullptr;
  MonitorElement *hRechitsEBGPUCPUratio = nullptr;
  MonitorElement *hRechitsEEGPUCPUratio = nullptr;
  MonitorElement *hRechitsEBdeltavsCPU = nullptr;
  MonitorElement *hRechitsEEdeltavsCPU = nullptr;

  // RecHits plots for EB and EE on both GPU and CPU
  MonitorElement *hSelectedRechitsEBGPU = nullptr;
  MonitorElement *hSelectedRechitsEBCPU = nullptr;
  MonitorElement *hSelectedRechitsEEGPU = nullptr;
  MonitorElement *hSelectedRechitsEECPU = nullptr;
  MonitorElement *hSelectedRechitsEBGPUvsCPU = nullptr;
  MonitorElement *hSelectedRechitsEEGPUvsCPU = nullptr;
  MonitorElement *hSelectedRechitsEBGPUCPUratio = nullptr;
  MonitorElement *hSelectedRechitsEEGPUCPUratio = nullptr;
  MonitorElement *hSelectedRechitsEBdeltavsCPU = nullptr;
  MonitorElement *hSelectedRechitsEEdeltavsCPU = nullptr;
  
  // RecHits plots for EB and EE on both GPU and CPU
  MonitorElement *hPositiveRechitsEBGPU = nullptr;
  MonitorElement *hPositiveRechitsEBCPU = nullptr;
  MonitorElement *hPositiveRechitsEEGPU = nullptr;
  MonitorElement *hPositiveRechitsEECPU = nullptr;
  MonitorElement *hPositiveRechitsEBGPUvsCPU = nullptr;
  MonitorElement *hPositiveRechitsEEGPUvsCPU = nullptr;
  MonitorElement *hPositiveRechitsEBGPUCPUratio = nullptr;
  MonitorElement *hPositiveRechitsEEGPUCPUratio = nullptr;
  MonitorElement *hPositiveRechitsEBdeltavsCPU = nullptr;
  MonitorElement *hPositiveRechitsEEdeltavsCPU = nullptr;

  // Energies plots for EB and EE on both GPU and CPU
  MonitorElement *hEnergiesEBGPU = nullptr;
  MonitorElement *hEnergiesEEGPU = nullptr;
  MonitorElement *hEnergiesEBCPU = nullptr;
  MonitorElement *hEnergiesEECPU = nullptr;
  MonitorElement *hEnergiesEBGPUvsCPU = nullptr;
  MonitorElement *hEnergiesEEGPUvsCPU = nullptr;
  MonitorElement *hEnergiesEBGPUCPUratio = nullptr;
  MonitorElement *hEnergiesEEGPUCPUratio = nullptr;
  MonitorElement *hEnergiesEBdeltavsCPU = nullptr;
  MonitorElement *hEnergiesEEdeltavsCPU = nullptr;

  // Chi2 plots for EB and EE on both GPU and CPU
  MonitorElement *hChi2EBGPU = nullptr;
  MonitorElement *hChi2EEGPU = nullptr;
  MonitorElement *hChi2EBCPU = nullptr;
  MonitorElement *hChi2EECPU = nullptr;
  MonitorElement *hChi2EBGPUvsCPU = nullptr;
  MonitorElement *hChi2EEGPUvsCPU = nullptr;
  MonitorElement *hChi2EBGPUCPUratio = nullptr;
  MonitorElement *hChi2EEGPUCPUratio = nullptr;
  MonitorElement *hChi2EBdeltavsCPU = nullptr;
  MonitorElement *hChi2EEdeltavsCPU = nullptr;

  // Flags plots for EB and EE on both GPU and CPU
  MonitorElement *hFlagsEBGPU = nullptr;
  MonitorElement *hFlagsEBCPU = nullptr;
  MonitorElement *hFlagsEEGPU = nullptr;
  MonitorElement *hFlagsEECPU = nullptr;
  MonitorElement *hFlagsEBGPUvsCPU = nullptr;
  MonitorElement *hFlagsEEGPUvsCPU = nullptr;
  MonitorElement *hFlagsEBGPUCPUratio = nullptr;
  MonitorElement *hFlagsEEGPUCPUratio = nullptr;
  MonitorElement *hFlagsEBdeltavsCPU = nullptr;
  MonitorElement *hFlagsEEdeltavsCPU = nullptr;

  // Extras plots for EB and EE on both GPU and CPU
  MonitorElement *hExtrasEBGPU = nullptr;
  MonitorElement *hExtrasEBCPU = nullptr;
  MonitorElement *hExtrasEEGPU = nullptr;
  MonitorElement *hExtrasEECPU = nullptr;
  MonitorElement *hExtrasEBGPUvsCPU = nullptr;
  MonitorElement *hExtrasEEGPUvsCPU = nullptr;
  MonitorElement *hExtrasEBGPUCPUratio = nullptr;
  MonitorElement *hExtrasEEGPUCPUratio = nullptr;
  MonitorElement *hExtrasEBdeltavsCPU = nullptr;
  MonitorElement *hExtrasEEdeltavsCPU = nullptr;
  
  static constexpr int nbins=200, last=5000;
  static constexpr int nbins_energy=300, last_energy=2;
  static constexpr int nbins_chi2=200, last_chi2=100;
  static constexpr int nbins_flag=40, last_flag=1500;
  static constexpr int nbins_extra=200, last_extra = 200;
  static constexpr int nbins_delta = 201; // use an odd number to center around 0
  static constexpr float delta = 0.2;
};

EcalRecHitGPUAnalyzer::EcalRecHitGPUAnalyzer(const edm::ParameterSet &parameters)
  : cpuEBTok_( consumes<InputProductEBCPU>(parameters.getParameter<edm::InputTag>("cpuEBRecHitCollection")) ),
    cpuEETok_( consumes<InputProductEECPU>(parameters.getParameter<edm::InputTag>("cpuEERecHitCollection")) ),
    gpuEBTok_( consumes<InputProductGPU>(parameters.getParameter<edm::InputTag>("gpuEBRecHitCollection")) ),
    gpuEETok_( consumes<InputProductGPU>(parameters.getParameter<edm::InputTag>("gpuEERecHitCollection")) ) {}

// ------------ method called for each event  ------------
void EcalRecHitGPUAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const auto &hitsEBcpu = iEvent.get(cpuEBTok_);
  const auto &hitsEEcpu = iEvent.get(cpuEETok_);
  const auto &hitsEBgpu = iEvent.get(gpuEBTok_);
  const auto &hitsEEgpu = iEvent.get(gpuEETok_);

  const auto cpu_eb_size = hitsEBcpu.size();
  const auto cpu_ee_size = hitsEEcpu.size();
  auto gpu_eb_size = hitsEBgpu.energy.size();
  auto gpu_ee_size = hitsEEgpu.energy.size();

  if (cpu_eb_size!=gpu_eb_size or cpu_ee_size!=gpu_ee_size)
    edm::LogError("DifferentSizes_BEFORE_Selection") <<
      "  EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << gpu_eb_size <<
      " (gpu)\n" <<
      "  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << gpu_ee_size <<
      " (gpu)" << std::endl;

  const float eb_ratio = (float)gpu_eb_size / cpu_eb_size;
  const float ee_ratio = (float)gpu_ee_size / cpu_ee_size;

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

  unsigned selected_gpu_eb_size = 0;
  unsigned selected_gpu_ee_size = 0;
  unsigned positive_gpu_eb_size = 0;
  unsigned positive_gpu_ee_size = 0;

  // EB:
  for (unsigned i=0; i<gpu_eb_size; ++i) {
    const auto did_gpu = hitsEBgpu.did[i];  // set the did for the current RecHit
    const auto enr_gpu = hitsEBgpu.energy[i];
    const auto chi2_gpu = hitsEBgpu.chi2[i];
    const auto flag_gpu = hitsEBgpu.flagBits[i];
    const auto extra_gpu = hitsEBgpu.extra[i];

    // you have "-1" if the crystal is not selected
    if (enr_gpu >= 0) {
      selected_gpu_eb_size++;

      if (enr_gpu > 0) {
	positive_gpu_eb_size++;
      }

      // find the Rechit on CPU reflecting the same did
      const auto cpu_iter = hitsEBcpu.find(DetId{did_gpu});
      if (cpu_iter == hitsEBcpu.end()) {
	edm::LogError("MissingDetId") << "Did not find DetId " << did_gpu << " in a CPU collection\n";
	continue;
      }
      // Set the variables for CPU
      auto const enr_cpu = cpu_iter->energy();
      auto const chi2_cpu = cpu_iter->chi2();
      //         auto const flag_cpu = cpu_iter->flagBits();
      auto const flag_cpu = 1;
      //         auto const extra_cpu = cpu_iter->extra();
      auto const extra_cpu = 1;
      //       auto const flag_cpu = cpu_iter->flagBits() ? cpu_iter->flagBits():-1;
      //       auto const extra_cpu = cpu_iter->extra() ? cpu_iter->extra():-1;

      // Fill the energy and Chi2 histograms for GPU and CPU and their comparisons with delta
      hEnergiesEBGPU->Fill(enr_gpu);
      hEnergiesEBCPU->Fill(enr_cpu);
      hEnergiesEBGPUvsCPU->Fill(enr_cpu, enr_gpu);
      hEnergiesEBGPUCPUratio->Fill(enr_gpu / enr_cpu);
      hEnergiesEBdeltavsCPU->Fill(enr_cpu, enr_gpu - enr_cpu);

      hChi2EBGPU->Fill(chi2_gpu);
      hChi2EBCPU->Fill(chi2_cpu);
      hChi2EBGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EBGPUCPUratio->Fill(chi2_gpu / chi2_cpu);
      hChi2EBdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

      hFlagsEBGPU->Fill(flag_gpu);
      hFlagsEBCPU->Fill(flag_cpu);
      hFlagsEBGPUvsCPU->Fill(flag_cpu, flag_gpu);
      hFlagsEBGPUCPUratio->Fill(flag_cpu ? flag_gpu / flag_cpu : -1);
      hFlagsEBdeltavsCPU->Fill(flag_cpu, flag_gpu - flag_cpu);

      hExtrasEBGPU->Fill(extra_gpu);
      hExtrasEBCPU->Fill(extra_cpu);
      hExtrasEBGPUvsCPU->Fill(extra_cpu, extra_gpu);
      hExtrasEBGPUCPUratio->Fill(extra_cpu ? extra_gpu / extra_cpu : -1);
      hExtrasEBdeltavsCPU->Fill(extra_cpu, extra_gpu - extra_cpu);
    }
  }

  // EE:
  for (unsigned i=0; i<gpu_ee_size; ++i) {
    const auto did_gpu = hitsEEgpu.did[i];  // set the did for the current RecHit
    // Set the variables for GPU
    const auto enr_gpu = hitsEEgpu.energy[i];
    const auto chi2_gpu = hitsEEgpu.chi2[i];
    const auto flag_gpu = hitsEEgpu.flagBits[i];
    const auto extra_gpu = hitsEEgpu.extra[i];

    // you have "-1" if the crystal is not selected
    if (enr_gpu >= 0) {
      selected_gpu_ee_size++;

      if (enr_gpu > 0) {
	positive_gpu_ee_size++;
      }

      // find the Rechit on CPU reflecting the same did
      auto const cpu_iter = hitsEEcpu.find(DetId{did_gpu});
      if (cpu_iter == hitsEEcpu.end()) {
	edm::LogError("MissingDetId") << "Did not find DetId " << did_gpu << " in a CPU collection\n";
	continue;
      }
      // Set the variables for CPU
      auto const enr_cpu = cpu_iter->energy();
      auto const chi2_cpu = cpu_iter->chi2();
      //         auto const flag_cpu = cpu_iter->flagBits();
      auto const flag_cpu = 1;
      //         auto const extra_cpu = cpu_iter->extra();
      auto const extra_cpu = 1;
      //       auto const flag_cpu = cpu_iter->flagBits()?cpu_iter->flagBits():-1;
      //       auto const extra_cpu = cpu_iter->extra()?cpu_iter->extra():-1;

      // Fill the energy and Chi2 histograms for GPU and CPU and their comparisons with delta
      hEnergiesEEGPU->Fill(enr_gpu);
      hEnergiesEECPU->Fill(enr_cpu);
      hEnergiesEEGPUvsCPU->Fill(enr_cpu, enr_gpu);
      hEnergiesEEGPUCPUratio->Fill(enr_gpu / enr_cpu);
      hEnergiesEEdeltavsCPU->Fill(enr_cpu, enr_gpu - enr_cpu);

      hChi2EEGPU->Fill(chi2_gpu);
      hChi2EECPU->Fill(chi2_cpu);
      hChi2EEGPUvsCPU->Fill(chi2_cpu, chi2_gpu);
      hChi2EEGPUCPUratio->Fill(chi2_gpu / chi2_cpu);
      hChi2EEdeltavsCPU->Fill(chi2_cpu, chi2_gpu - chi2_cpu);

      hFlagsEEGPU->Fill(flag_gpu);
      hFlagsEECPU->Fill(flag_cpu);
      hFlagsEEGPUvsCPU->Fill(flag_cpu, flag_gpu);
      hFlagsEEGPUCPUratio->Fill(flag_cpu ? flag_gpu / flag_cpu : -1);
      hFlagsEEdeltavsCPU->Fill(flag_cpu, flag_gpu - flag_cpu);

      hExtrasEEGPU->Fill(extra_gpu);
      hExtrasEECPU->Fill(extra_cpu);
      hExtrasEEGPUvsCPU->Fill(extra_cpu, extra_gpu);
      hExtrasEEGPUCPUratio->Fill(extra_cpu ? extra_gpu / extra_cpu : -1);
      hExtrasEEdeltavsCPU->Fill(extra_cpu, extra_gpu - extra_cpu);
    }
  }

  /////////////////
  //Rechit Counting
  /////////////////
  float selected_eb_ratio = static_cast<float>(selected_gpu_eb_size) / cpu_eb_size;
  float selected_ee_ratio = static_cast<float>(selected_gpu_ee_size) / cpu_ee_size;

  // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
  hSelectedRechitsEBGPU->Fill(selected_gpu_eb_size);
  hSelectedRechitsEBCPU->Fill(cpu_eb_size);
  hSelectedRechitsEEGPU->Fill(selected_gpu_ee_size);
  hSelectedRechitsEECPU->Fill(cpu_ee_size);
  hSelectedRechitsEBGPUvsCPU->Fill(cpu_eb_size, selected_gpu_eb_size);
  hSelectedRechitsEEGPUvsCPU->Fill(cpu_ee_size, selected_gpu_ee_size);
  hSelectedRechitsEBGPUCPUratio->Fill(selected_eb_ratio);
  hSelectedRechitsEEGPUCPUratio->Fill(selected_ee_ratio);
  hSelectedRechitsEBdeltavsCPU->Fill(cpu_eb_size, selected_gpu_eb_size - cpu_eb_size);
  hSelectedRechitsEEdeltavsCPU->Fill(cpu_ee_size, selected_gpu_ee_size - cpu_ee_size);

  unsigned positive_cpu_eb_size = 0;
  unsigned positive_cpu_ee_size = 0;

  // EB:
  for (unsigned i = 0; i<cpu_eb_size; ++i) {
    const auto enr_cpu = hitsEBcpu[i].energy();
    if (enr_cpu>0) {
      positive_cpu_eb_size++;
    }
  }
  // EE:
  for (unsigned i=0; i<cpu_ee_size; ++i) {
    auto const enr_cpu = hitsEEcpu[i].energy();
    if (enr_cpu > 0) {
      positive_cpu_ee_size++;
    }
  }

  float positive_eb_ratio = static_cast<float>(positive_gpu_eb_size) / positive_cpu_eb_size;
  float positive_ee_ratio = static_cast<float>(positive_gpu_ee_size) / positive_cpu_ee_size;

  // Filling up the histograms on events sizes for EB and EE on both GPU and CPU
  hPositiveRechitsEBGPU->Fill(positive_gpu_eb_size);
  hPositiveRechitsEBCPU->Fill(positive_cpu_eb_size);
  hPositiveRechitsEEGPU->Fill(positive_gpu_ee_size);
  hPositiveRechitsEECPU->Fill(positive_cpu_ee_size);
  hPositiveRechitsEBGPUvsCPU->Fill(positive_cpu_eb_size, positive_gpu_eb_size);
  hPositiveRechitsEEGPUvsCPU->Fill(positive_cpu_ee_size, positive_gpu_ee_size);
  hPositiveRechitsEBGPUCPUratio->Fill(positive_eb_ratio);
  hPositiveRechitsEEGPUCPUratio->Fill(positive_ee_ratio);
  hPositiveRechitsEBdeltavsCPU->Fill(positive_cpu_eb_size, positive_gpu_eb_size - positive_cpu_eb_size);
  hPositiveRechitsEEdeltavsCPU->Fill(positive_cpu_ee_size, positive_gpu_ee_size - positive_cpu_ee_size);

  if (cpu_eb_size != selected_gpu_eb_size or cpu_ee_size != selected_gpu_ee_size) {
    edm::LogError("DifferentSizes_AFTER_Selection") <<
      "  EB size: " << std::setw(4) << cpu_eb_size << " (cpu) vs " << std::setw(4) << selected_gpu_eb_size <<
      " (gpu)\n" <<
      "  EE size: " << std::setw(4) << cpu_ee_size << " (cpu) vs " << std::setw(4) << selected_gpu_ee_size <<
      " (gpu)" << std::endl;
  }
  if (cpu_eb_size < selected_gpu_eb_size or cpu_ee_size < selected_gpu_ee_size)
    edm::LogError("MissingRecHits_AFTER_Selection") << std::endl;
}

void EcalRecHitGPUAnalyzer::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  std::string logTraceName("EcalRecHitGPUAnalyzer");

  LogTrace(logTraceName) << "Parameters initialization";

  //iBooker.setCurrentFolder("DQM_tmp/");  // Use folder with name of PAG

  hRechitsEBGPU = iBooker.book1D("RechitsEBGPU_NoFilter", "RechitsEBGPU; No. of Rechits (no GPU filter)", nbins, 0, last);
  hRechitsEBCPU = iBooker.book1D("RechitsEBCPU_NoFilter", "RechitsEBCPU; No. of Rechits (no GPU filter)", nbins, 0, last);
  hRechitsEEGPU = iBooker.book1D("RechitsEEGPU_NoFilter", "RechitsEEGPU; No. of Rechits (no GPU filter)", nbins, 0, last);
  hRechitsEECPU = iBooker.book1D("RechitsEECPU_NoFilter", "RechitsEECPU; No. of Rechits (no GPU filter)", nbins, 0, last);
  hRechitsEBGPUvsCPU =
    iBooker.book2D("RechitsEBGPUvsCPU_NoFilter", "RechitsEBGPUvsCPU; CPU; GPU (no GPU filter)", last, 0, last, last, 0, last);
  hRechitsEEGPUvsCPU =
    iBooker.book2D("RechitsEEGPUvsCPU_NoFilter", "RechitsEEGPUvsCPU; CPU; GPU (no GPU filter)", last, 0, last, last, 0, last);
  hRechitsEBGPUCPUratio =
    iBooker.book1D("RechitsEBGPU/CPUratio_NoFilter", "RechitsEBGPU/CPUratio; GPU/CPU (no GPU filter)", 200, 0.95, 1.05);
  hRechitsEEGPUCPUratio =
    iBooker.book1D("RechitsEEGPU/CPUratio_NoFilter", "RechitsEEGPU/CPUratio; GPU/CPU (no GPU filter)", 200, 0.95, 1.05);
  hRechitsEBdeltavsCPU =
    iBooker.book2D("RechitsEBdeltavsCPU_NoFilter", "RechitsEBdeltavsCPU (no GPU filter)", nbins, 0, last, nbins_delta, -delta, delta);
  hRechitsEEdeltavsCPU =
    iBooker.book2D("RechitsEEdeltavsCPU_NoFilter", "RechitsEEdeltavsCPU (no GPU filter)", nbins, 0, last, nbins_delta, -delta, delta);

  // RecHits plots for EB and EE on both GPU and CPU
  hSelectedRechitsEBGPU = iBooker.book1D("RechitsEBGPU", "RechitsEBGPU; No. of Rechits", nbins, 0, last);
  hSelectedRechitsEBCPU = iBooker.book1D("RechitsEBCPU", "RechitsEBCPU; No. of Rechits", nbins, 0, last);
  hSelectedRechitsEEGPU = iBooker.book1D("RechitsEEGPU", "RechitsEEGPU; No. of Rechits", nbins, 0, last);
  hSelectedRechitsEECPU = iBooker.book1D("RechitsEECPU", "RechitsEECPU; No. of Rechits", nbins, 0, last);
  hSelectedRechitsEBGPUvsCPU =
    iBooker.book2D("RechitsEBGPUvsCPU", "RechitsEBGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  hSelectedRechitsEEGPUvsCPU =
    iBooker.book2D("RechitsEEGPUvsCPU", "RechitsEEGPUvsCPU; CPU; GPU", last, 0, last, last, 0, last);
  hSelectedRechitsEBGPUCPUratio =
    iBooker.book1D("RechitsEBGPU/CPUratio", "RechitsEBGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  hSelectedRechitsEEGPUCPUratio =
    iBooker.book1D("RechitsEEGPU/CPUratio", "RechitsEEGPU/CPUratio; GPU/CPU", 200, 0.95, 1.05);
  hSelectedRechitsEBdeltavsCPU =
    iBooker.book2D("RechitsEBdeltavsCPU", "RechitsEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  hSelectedRechitsEEdeltavsCPU =
    iBooker.book2D("RechitsEEdeltavsCPU", "RechitsEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  // RecHits plots for EB and EE on both GPU and CPU
  hPositiveRechitsEBGPU = iBooker.book1D("RechitsEBGPU_Positive", "RechitsEBGPU; No. of Rechits (positive)", nbins, 0, last);
  hPositiveRechitsEBCPU = iBooker.book1D("RechitsEBCPU_Positive", "RechitsEBCPU; No. of Rechits (positive)", nbins, 0, last);
  hPositiveRechitsEEGPU = iBooker.book1D("RechitsEEGPU_Positive", "RechitsEEGPU; No. of Rechits (positive)", nbins, 0, last);
  hPositiveRechitsEECPU = iBooker.book1D("RechitsEECPU_Positive", "RechitsEECPU; No. of Rechits (positive)", nbins, 0, last);
  hPositiveRechitsEBGPUvsCPU =
    iBooker.book2D("RechitsEBGPUvsCPU_Positive", "RechitsEBGPUvsCPU; CPU; GPU (positive)", last, 0, last, last, 0, last);
  hPositiveRechitsEEGPUvsCPU =
    iBooker.book2D("RechitsEEGPUvsCPU_Positive", "RechitsEEGPUvsCPU; CPU; GPU (positive)", last, 0, last, last, 0, last);
  hPositiveRechitsEBGPUCPUratio =
    iBooker.book1D("RechitsEBGPU/CPUratio_Positive", "RechitsEBGPU/CPUratio; GPU/CPU (positive)", 200, 0.95, 1.05);
  hPositiveRechitsEEGPUCPUratio =
    iBooker.book1D("RechitsEEGPU/CPUratio_Positive", "RechitsEEGPU/CPUratio; GPU/CPU (positive)", 200, 0.95, 1.05);
  hPositiveRechitsEBdeltavsCPU =
    iBooker.book2D("RechitsEBdeltavsCPU_Positive", "RechitsEBdeltavsCPU (positive)", nbins, 0, last, nbins_delta, -delta, delta);
  hPositiveRechitsEEdeltavsCPU =
    iBooker.book2D("RechitsEEdeltavsCPU_Positive", "RechitsEEdeltavsCPU (positive)", nbins, 0, last, nbins_delta, -delta, delta);

  // Energies plots for EB and EE on both GPU and CPU
  hEnergiesEBGPU = iBooker.book1D("EnergiesEBGPU", "EnergiesEBGPU; Energy [GeV]", nbins_energy, 0, last_energy);
  hEnergiesEEGPU = iBooker.book1D("EnergiesEEGPU", "EnergiesEEGPU; Energy [GeV]", nbins_energy, 0, last_energy);
  hEnergiesEBCPU = iBooker.book1D("EnergiesEBCPU", "EnergiesEBCPU; Energy [GeV]", nbins_energy, 0, last_energy);
  hEnergiesEECPU = iBooker.book1D("EnergiesEECPU", "EnergiesEECPU; Energy [GeV]", nbins_energy, 0, last_energy);
  hEnergiesEBGPUvsCPU = iBooker.book2D(
				       "EnergiesEBGPUvsCPU", "EnergiesEBGPUvsCPU; CPU; GPU", nbins_energy, 0, last_energy, nbins_energy, 0, last_energy);
  hEnergiesEEGPUvsCPU = iBooker.book2D(
				       "EnergiesEEGPUvsCPU", "EnergiesEEGPUvsCPU; CPU; GPU", nbins_energy, 0, last_energy, nbins_energy, 0, last_energy);
  hEnergiesEBGPUCPUratio = iBooker.book1D("EnergiesEBGPU/CPUratio", "EnergiesEBGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  hEnergiesEEGPUCPUratio = iBooker.book1D("EnergiesEEGPU/CPUratio", "EnergiesEEGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  hEnergiesEBdeltavsCPU =
    iBooker.book2D("EnergiesEBdeltavsCPU", "EnergiesEBdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);
  hEnergiesEEdeltavsCPU =
    iBooker.book2D("EnergiesEEdeltavsCPU", "EnergiesEEdeltavsCPU", nbins, 0, last, nbins_delta, -delta, delta);

  // Chi2 plots for EB and EE on both GPU and CPU
  hChi2EBGPU = iBooker.book1D("Chi2EBGPU", "Chi2EBGPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  hChi2EEGPU = iBooker.book1D("Chi2EEGPU", "Chi2EEGPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  hChi2EBCPU = iBooker.book1D("Chi2EBCPU", "Chi2EBCPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  hChi2EECPU = iBooker.book1D("Chi2EECPU", "Chi2EECPU; Ch^{2}", nbins_chi2, 0, last_chi2);
  hChi2EBGPUvsCPU = iBooker.book2D("Chi2EBGPUvsCPU", "Chi2EBGPUvsCPU; CPU; GPU", nbins_chi2, 0, 100, nbins_chi2, 0, 100);
  hChi2EEGPUvsCPU = iBooker.book2D("Chi2EEGPUvsCPU", "Chi2EEGPUvsCPU; CPU; GPU", nbins_chi2, 0, 100, nbins_chi2, 0, 100);
  hChi2EBGPUCPUratio = iBooker.book1D("Chi2EBGPU/CPUratio", "Chi2EBGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  hChi2EEGPUCPUratio = iBooker.book1D("Chi2EEGPU/CPUratio", "Chi2EEGPU/CPUratio; GPU/CPU", 100, 0.8, 1.2);
  hChi2EBdeltavsCPU =
    iBooker.book2D("Chi2EBdeltavsCPU", "Chi2EBdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);
  hChi2EEdeltavsCPU =
    iBooker.book2D("Chi2EEdeltavsCPU", "Chi2EEdeltavsCPU", nbins_chi2, 0, last_chi2, nbins_delta, -delta, delta);

  // Flags plots for EB and EE on both GPU and CPU
  hFlagsEBGPU = iBooker.book1D("FlagsEBGPU", "FlagsEBGPU; Flags", nbins_flag, -10, last_flag);
  hFlagsEBCPU = iBooker.book1D("FlagsEBCPU", "FlagsEBCPU; Flags", nbins_flag, -10, last_flag);
  hFlagsEEGPU = iBooker.book1D("FlagsEEGPU", "FlagsEEGPU; Flags", nbins_flag, -10, last_flag);
  hFlagsEECPU = iBooker.book1D("FlagsEECPU", "FlagsEECPU; Flags", nbins_flag, -10, last_flag);
  hFlagsEBGPUvsCPU =
    iBooker.book2D("FlagsEBGPUvsCPU", "FlagsEBGPUvsCPU; CPU; GPU", nbins_flag, -10, last_flag, nbins_flag, -10, last_flag);
  hFlagsEEGPUvsCPU =
    iBooker.book2D("FlagsEEGPUvsCPU", "FlagsEEGPUvsCPU; CPU; GPU", nbins_flag, -10, last_flag, nbins_flag, -10, last_flag);
  hFlagsEBGPUCPUratio = iBooker.book1D("FlagsEBGPU/CPUratio", "FlagsEBGPU/CPUratio; GPU/CPU", 50, -5, 10);
  hFlagsEEGPUCPUratio = iBooker.book1D("FlagsEEGPU/CPUratio", "FlagsEEGPU/CPUratio; GPU/CPU", 50, -5, 10);
  hFlagsEBdeltavsCPU =
    iBooker.book2D("FlagsEBdeltavsCPU", "FlagsEBdeltavsCPU", nbins_flag, -10, last_flag, nbins_delta, -delta, delta);
  hFlagsEEdeltavsCPU =
    iBooker.book2D("FlagsEEdeltavsCPU", "FlagsEEdeltavsCPU", nbins_flag, -10, last_flag, nbins_delta, -delta, delta);

  // Extras plots for EB and EE on both GPU and CPU
  hExtrasEBGPU = iBooker.book1D("ExtrasEBGPU", "ExtrasEBGPU; No. of Extras", nbins_extra, 0, last_extra);
  hExtrasEBCPU = iBooker.book1D("ExtrasEBCPU", "ExtrasEBCPU; No. of Extras", nbins_extra, 0, last_extra);
  hExtrasEEGPU = iBooker.book1D("ExtrasEEGPU", "ExtrasEEGPU; No. of Extras", nbins_extra, 0, last_extra);
  hExtrasEECPU = iBooker.book1D("ExtrasEECPU", "ExtrasEECPU; No. of Extras", nbins_extra, 0, last_extra);
  hExtrasEBGPUvsCPU = iBooker.book2D(
				     "ExtrasEBGPUvsCPU", "ExtrasEBGPUvsCPU; CPU; GPU", nbins_extra, 0, last_extra, nbins_extra, 0, last_extra);
  hExtrasEEGPUvsCPU = iBooker.book2D(
				     "ExtrasEEGPUvsCPU", "ExtrasEEGPUvsCPU; CPU; GPU", nbins_extra, 0, last_extra, nbins_extra, 0, last_extra);
  hExtrasEBGPUCPUratio = iBooker.book1D("ExtrasEBGPU/CPUratio", "ExtrasEBGPU/CPUratio; GPU/CPU", 50, 0.0, 2.0);
  hExtrasEEGPUCPUratio = iBooker.book1D("ExtrasEEGPU/CPUratio", "ExtrasEEGPU/CPUratio; GPU/CPU", 50, 0.0, 2.0);
  hExtrasEBdeltavsCPU =
    iBooker.book2D("ExtrasEBdeltavsCPU", "ExtrasEBdeltavsCPU", nbins_extra, 0, last_extra, nbins_delta, -delta, delta);
  hExtrasEEdeltavsCPU =
    iBooker.book2D("ExtrasEEdeltavsCPU", "ExtrasEEdeltavsCPU", nbins_extra, 0, last_extra, nbins_delta, -delta, delta);
}

DEFINE_FWK_MODULE(EcalRecHitGPUAnalyzer);
