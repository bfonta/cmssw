#include "HeterogeneousHGCalHEFRecHitProducer.h"

HeterogeneousHGCalHEFRecHitProducer::HeterogeneousHGCalHEFRecHitProducer(const edm::ParameterSet& ps):
  token_(consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("HGCHEFUncalibRecHitsTok")))
{
  cdata_.hgcHEF_keV2DIGI_   = ps.getParameter<double>("HGCHEF_keV2DIGI");
  cdata_.xmin_              = ps.getParameter<double>("minValSiPar"); //float
  cdata_.xmax_              = ps.getParameter<double>("maxValSiPar"); //float
  cdata_.aterm_             = ps.getParameter<double>("constSiPar"); //float
  cdata_.cterm_             = ps.getParameter<double>("noiseSiPar"); //float
  vdata_.fCPerMIP_          = ps.getParameter< std::vector<double> >("HGCHEF_fCPerMIP");
  vdata_.cce_               = ps.getParameter<edm::ParameterSet>("HGCHEF_cce").getParameter<std::vector<double> >("values");
  vdata_.noise_fC_          = ps.getParameter<edm::ParameterSet>("HGCHEF_noise_fC").getParameter<std::vector<double> >("values");
  vdata_.rcorr_             = ps.getParameter< std::vector<double> >("rcorr");
  vdata_.weights_           = ps.getParameter< std::vector<double> >("weights");
  cdata_.fhOffset_          = ps.getParameter<uint32_t>("offset"); //ddd_->layers(true);
  cdata_.s_hgcHEF_fCPerMIP_ = vdata_.fCPerMIP_.size();
  cdata_.s_hgcHEF_cce_      = vdata_.cce_.size();
  cdata_.s_hgcHEF_noise_fC_ = vdata_.noise_fC_.size();
  cdata_.s_rcorr_           = vdata_.rcorr_.size();
  cdata_.s_weights_         = vdata_.weights_.size();
  cdata_.hgchefUncalib2GeV_ = 1e-6 / cdata_.hgcHEF_keV2DIGI_;
  vdata_.waferTypeL_        = {0, 1, 2};//ddd_->retWaferTypeL(); if depends on geometry the allocation is tricky!
  cdata_.s_waferTypeL_      = vdata_.waferTypeL_.size();

  tools_.reset(new hgcal::RecHitTools());

  produces<HGChefRecHitCollection>(collection_name_);
}

HeterogeneousHGCalHEFRecHitProducer::~HeterogeneousHGCalHEFRecHitProducer()
{
  delete kmdata_;
  delete h_kcdata_;
  delete d_kcdata_;
  delete old_soa_;
  delete d_oldhits_;
  delete d_newhits_;
  delete d_newhits_final_;
  delete h_newhits_;

  delete[] h_conds_.layer;
}

void HeterogeneousHGCalHEFRecHitProducer::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder w) {
  const cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};
  set_geometry_(setup);

  event.getByToken(token_, handle_hef_);
  const auto &hits_hef = *handle_hef_;

  unsigned int nhits = hits_hef.size();
  stride_ = ( (nhits-1)/32 + 1 ) * 32; //align to warp boundary
  allocate_memory_();

  std::cout << "check conditions" << std::endl;

  set_conditions(h_conds_, nhits, stride_, hits_hef);
  HeterogeneousConditionsESProductWrapper esproduct(nhits, stride_, h_conds_);
  const HeterogeneousConditionsESProduct* d_conds = esproduct.getHeterogeneousConditionsESProductAsync(ctx.stream());

  std::cout << "conver constant data" << std::endl;
  convert_constant_data_(h_kcdata_);

  convert_collection_data_to_soa_(hits_hef, old_soa_, nhits);

  std::cout << "kernel manager" << std::endl;
  kmdata_ = new KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>(nhits, stride_, old_soa_, d_oldhits_, d_newhits_, d_newhits_final_, h_newhits_);
  KernelManagerHGCalRecHit kernel_manager(kmdata_, d_conds);
  kernel_manager.run_kernels(h_kcdata_, d_kcdata_);

  std::cout << "last" << std::endl;
  rechits_ = std::make_unique<HGCRecHitCollection>();
  convert_soa_data_to_collection_(*rechits_, h_newhits_, nhits);
}

void HeterogeneousHGCalHEFRecHitProducer::set_conditions(HeterogeneousConditionsESProduct& c, const unsigned int& nelems, const unsigned int& stride, const HGChefUncalibratedRecHitCollection& hits) {
  c.layer = new int[2*stride];
  c.wafer = c.layer + stride;
  for(unsigned int i=0; i<nelems; ++i)
    {
      HGCalDetId obj( hits[i].id().rawId() );
      c.layer[i] = obj.layer();
      c.wafer[i] = obj.wafer();
    }
}

void HeterogeneousHGCalHEFRecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
  cms::cuda::ScopedContextProduce ctx{ctxState_}; //only for GPU to GPU producers
  event.put(std::move(rechits_), collection_name_);
}

void HeterogeneousHGCalHEFRecHitProducer::allocate_memory_()
{
  old_soa_ = new HGCUncalibratedRecHitSoA();
  d_oldhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_ = new HGCUncalibratedRecHitSoA();
  d_newhits_final_ = new HGCRecHitSoA();
  h_newhits_ = new HGCRecHitSoA();
  h_kcdata_ = new KernelConstantData<HGChefUncalibratedRecHitConstantData>(cdata_, vdata_);
  d_kcdata_ = new KernelConstantData<HGChefUncalibratedRecHitConstantData>(cdata_, vdata_);

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

void HeterogeneousHGCalHEFRecHitProducer::set_geometry_(const edm::EventSetup& setup)
{
  tools_->getEventSetup(setup);
  std::string handle_str;
  handle_str = "HGCalHESiliconSensitive";
  edm::ESHandle<HGCalGeometry> handle;
  setup.get<IdealGeometryRecord>().get(handle_str, handle);
  //ddd_ = &(handle->topology().dddConstants());
  //cdata_.fhOffset_ = ddd_->layers(true); see RecoLocalCalo/HGCalRecAlgos/src/RecHitTools.cc
}

void HeterogeneousHGCalHEFRecHitProducer::convert_constant_data_(KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata)
{
  for(int i=0; i<kcdata->data_.s_hgcHEF_fCPerMIP_; ++i)
    kcdata->data_.hgcHEF_fCPerMIP_[i] = kcdata->vdata_.fCPerMIP_[i];
  for(int i=0; i<kcdata->data_.s_hgcHEF_cce_; ++i)
    kcdata->data_.hgcHEF_cce_[i] = kcdata->vdata_.cce_[i];
  for(int i=0; i<kcdata->data_.s_hgcHEF_noise_fC_; ++i)
    kcdata->data_.hgcHEF_noise_fC_[i] = kcdata->vdata_.noise_fC_[i];
  for(int i=0; i<kcdata->data_.s_rcorr_; ++i)
    kcdata->data_.rcorr_[i] = kcdata->vdata_.rcorr_[i];
  for(int i=0; i<kcdata->data_.s_weights_; ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
  for(int i=0; i<kcdata->data_.s_waferTypeL_; ++i)
    kcdata->data_.waferTypeL_[i] = kcdata->vdata_.waferTypeL_[i];
}

void HeterogeneousHGCalHEFRecHitProducer::convert_collection_data_to_soa_(const edm::SortedCollection<HGCUncalibratedRecHit>& hits, HGCUncalibratedRecHitSoA* d, const unsigned int& nhits)
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
      d->wafer_[i] = 1; //CHANGE!!! use the geometry
      d->layer_[i] = 1; //CHANGE!!! use the geometry
    }
}

void HeterogeneousHGCalHEFRecHitProducer::convert_soa_data_to_collection_(HGCRecHitCollection& rechits, HGCRecHitSoA *d, const unsigned int& nhits)
{
  rechits.reserve(nhits);
  for(uint i=0; i<nhits; ++i)
    {
      DetId id_converted( d->id_[i] );
      rechits.emplace_back( HGCRecHit(id_converted, d->energy_[i], d->time_[i], 0, d->flagBits_[i]) );
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HeterogeneousHGCalHEFRecHitProducer);
