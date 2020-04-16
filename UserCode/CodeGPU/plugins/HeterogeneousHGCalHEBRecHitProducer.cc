#include "HeterogeneousHGCalHEBRecHitProducer.h"

HeterogeneousHGCalHEBRecHitProducer::HeterogeneousHGCalHEBRecHitProducer(const edm::ParameterSet& ps):
  token_(consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("HGCHEBUncalibRecHitsTok")))
{
  cdata_.hgcHEB_keV2DIGI_   = ps.getParameter<double>("HGCHEB_keV2DIGI");
  cdata_.hgcHEB_noise_MIP_  = ps.getParameter<edm::ParameterSet>("HGCHEB_noise_MIP").getParameter<double>("noise_MIP");
  vdata_.weights_           = ps.getParameter< std::vector<double> >("weights");
  cdata_.bhOffset_          = ps.getParameter<uint32_t>("offset"); //ddd_->layers(true);
  cdata_.s_weights_         = vdata_.weights_.size();
  cdata_.hgchebUncalib2GeV_ = 1e-6 / cdata_.hgcHEB_keV2DIGI_;

  tools_.reset(new hgcal::RecHitTools());

  produces<HGChebRecHitCollection>(collection_name_);
}

HeterogeneousHGCalHEBRecHitProducer::~HeterogeneousHGCalHEBRecHitProducer()
{
  delete kmdata_;
  delete h_kcdata_;
  delete d_kcdata_;
  delete old_soa_;
  delete d_oldhits_;
  delete d_newhits_;
  delete d_newhits_final_;
  delete h_newhits_;
}

void HeterogeneousHGCalHEBRecHitProducer::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder w) {
  const cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};
  set_geometry_(setup);
  
  event.getByToken(token_, handle_heb_);
  const auto &hits_heb = *handle_heb_;

  unsigned int nhits = hits_heb.size();
  stride_ = ( (nhits-1)/32 + 1 ) * 32; //align to warp boundary
  allocate_memory_();
  convert_constant_data_(h_kcdata_);

  convert_collection_data_to_soa_(hits_heb, old_soa_, nhits);

  kmdata_ = new KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>(nhits, stride_, old_soa_, d_oldhits_, d_newhits_, d_newhits_final_, h_newhits_);
  KernelManagerHGCalRecHit kernel_manager(kmdata_);
  kernel_manager.run_kernels(h_kcdata_, d_kcdata_);

  rechits_ = std::make_unique<HGCRecHitCollection>();
  convert_soa_data_to_collection_(*rechits_, h_newhits_, nhits);
}

void HeterogeneousHGCalHEBRecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
  cms::cuda::ScopedContextProduce ctx{ctxState_}; //only for GPU to GPU producers
  event.put(std::move(rechits_), collection_name_);
}

void HeterogeneousHGCalHEBRecHitProducer::allocate_memory_()
{
  old_soa_ = new HGCUncalibratedRecHitSoA();
  d_oldhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_final_ = new HGCRecHitSoA();
  h_newhits_ = new HGCRecHitSoA();
  h_kcdata_ = new KernelConstantData<HGChebUncalibratedRecHitConstantData>(cdata_, vdata_);
  d_kcdata_ = new KernelConstantData<HGChebUncalibratedRecHitConstantData>(cdata_, vdata_);

  //_allocate pinned memory for constants on the host
  memory::allocation::host(h_kcdata_, h_mem_const_);
  //_allocate pinned memory for constants on the device
  memory::allocation::device(d_kcdata_, d_mem_const_);
  //_allocate memory for hits on the host
  memory::allocation::host(stride_, old_soa_, h_mem_in_);
  //_allocate memory for hits on the device
  memory::allocation::device(stride_, d_oldhits_, d_newhits_, d_newhits_final_, d_mem_);
  //_allocate memory for hits on the host
  memory::allocation::host(stride_, h_newhits_, h_mem_out_);
}

void HeterogeneousHGCalHEBRecHitProducer::set_geometry_(const edm::EventSetup& setup)
{
  tools_->getEventSetup(setup);
  std::string handle_str;
  handle_str = "HGCalHEScintillatorSensitive";
  edm::ESHandle<HGCalGeometry> handle;
  setup.get<IdealGeometryRecord>().get(handle_str, handle);
  //ddd_ = &(handle->topology().dddConstants());
  //cdata_.bhOffset_ = fhOffset + ddd_->layers(true); see RecoLocalCalo/HGCalRecAlgos/src/RecHitTools.cc
}

void HeterogeneousHGCalHEBRecHitProducer::convert_constant_data_(KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata)
{
  for(int i=0; i<kcdata->data_.s_weights_; ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

void HeterogeneousHGCalHEBRecHitProducer::convert_collection_data_to_soa_(const edm::SortedCollection<HGCUncalibratedRecHit>& hits, HGCUncalibratedRecHitSoA* d, const unsigned int& nhits)
{
  for(unsigned int i=0; i<nhits; ++i)
    {
      d->amplitude_[i] = hits[i].amplitude();
      d->pedestal_[i] = hits[i].pedestal();
      d->jitter_[i] = hits[i].jitter();
      d->chi2_[i] = hits[i].chi2();
      d->OOTamplitude_[i] = hits[i].outOfTimeEnergy();
      d->OOTchi2_[i] = hits[i].outOfTimeChi2();
      d->flags_[i] = hits[i].flags();
      d->aux_[i] = 0;
      d->id_[i] = hits[i].id().rawId();
    }
}

void HeterogeneousHGCalHEBRecHitProducer::convert_soa_data_to_collection_(HGCRecHitCollection& rechits, HGCRecHitSoA *d, const unsigned int& nhits)
{
  rechits.reserve(nhits);
  for(uint i=0; i<nhits; ++i)
    {
      DetId id_converted( d->id_[i] );
      rechits.emplace_back( HGCRecHit(id_converted, d->energy_[i], d->time_[i], 0, d->flagBits_[i]) );
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HeterogeneousHGCalHEBRecHitProducer);
