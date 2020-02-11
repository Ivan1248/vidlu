graph [
  version 2
  directed 1
  charset "utf-8"
  rankdir "BT"
  node [
    id 0
    name "0"
    label "{AAEDecoder|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AAEDecoder|\l|}"
    ]
  ]
  node [
    id 1
    name "1"
    label "{AAEDiscriminator|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AAEDiscriminator|\l|}"
    ]
  ]
  node [
    id 2
    name "2"
    label "{AAEEncoder|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AAEEncoder|\l|}"
    ]
  ]
  node [
    id 3
    name "3"
    label "{AccumulatingMetric|\l|compute()\lreset()\lupdate()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AccumulatingMetric|\l|compute()\lreset()\lupdate()\l}"
    ]
  ]
  node [
    id 4
    name "4"
    label "{AdaptiveAvgPool2d|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AdaptiveAvgPool2d|\l|forward()\l}"
    ]
  ]
  node [
    id 5
    name "5"
    label "{AdversarialAutoencoder|discriminator\lprior_rand\l|discriminate_z()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AdversarialAutoencoder|discriminator\lprior_rand\l|discriminate_z()\l}"
    ]
  ]
  node [
    id 6
    name "6"
    label "{AdversarialSupervisedTrainer|attack\l|train_batch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AdversarialSupervisedTrainer|attack\l|train_batch()\l}"
    ]
  ]
  node [
    id 7
    name "7"
    label "{AdversarialTrainer|attack\lattack\lattack_f\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AdversarialTrainer|attack\lattack\lattack_f\l|}"
    ]
  ]
  node [
    id 8
    name "8"
    label "{AffineCoupling|scale\ltranslate\l|forward()\linvert()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AffineCoupling|scale\ltranslate\l|forward()\linvert()\l}"
    ]
  ]
  node [
    id 9
    name "9"
    label "{Attack|clip_bounds : NoneType\lget_predicted_label\lis_success\lloss_fn\lmodel\l|perturb()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Attack|clip_bounds : NoneType\lget_predicted_label\lis_success\lloss_fn\lmodel\l|perturb()\l}"
    ]
  ]
  node [
    id 10
    name "10"
    label "{Autoencoder|decoder\lencoder\l|decode()\lencode()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Autoencoder|decoder\lencoder\l|decode()\lencode()\l}"
    ]
  ]
  node [
    id 11
    name "11"
    label "{AutoencoderTrainer|\l|train_batch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AutoencoderTrainer|\l|train_batch()\l}"
    ]
  ]
  node [
    id 12
    name "12"
    label "{AvgPool|orig\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AvgPool|orig\l|build()\l}"
    ]
  ]
  node [
    id 13
    name "13"
    label "{AvgPool2d|ceil_mode : bool\lcount_include_pad : bool\lkernel_size\lpadding : int\lstride\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{AvgPool2d|ceil_mode : bool\lcount_include_pad : bool\lkernel_size\lpadding : int\lstride\l|forward()\l}"
    ]
  ]
  node [
    id 14
    name "14"
    label "{BagNet|avg_pool : bool\lavgpool\lblock\lbn1\lfc\linplanes : int\llayer1\llayer2\llayer3\llayer4\lrelu\lroot\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BagNet|avg_pool : bool\lavgpool\lblock\lbn1\lfc\linplanes : int\llayer1\llayer2\llayer3\llayer4\lrelu\lroot\l|forward()\l}"
    ]
  ]
  node [
    id 15
    name "15"
    label "{BasicBlock|bn1\lbn2\lconv1\lconv2\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BasicBlock|bn1\lbn2\lconv1\lconv2\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    ]
  ]
  node [
    id 16
    name "16"
    label "{BatchDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BatchDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 17
    name "17"
    label "{BatchNorm|orig : NoneType\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BatchNorm|orig : NoneType\l|build()\l}"
    ]
  ]
  node [
    id 18
    name "18"
    label "{BatchNorm2d|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BatchNorm2d|\l|}"
    ]
  ]
  node [
    id 19
    name "19"
    label "{Bottleneck|bn1\lbn2\lbn3\lconv1\lconv2\lconv3\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Bottleneck|bn1\lbn2\lbn3\lconv1\lconv2\lconv3\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    ]
  ]
  node [
    id 20
    name "20"
    label "{BranchOut|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BranchOut|\l|forward()\l}"
    ]
  ]
  node [
    id 21
    name "21"
    label "{BuildableModExt|args\l|build()\lpost_build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{BuildableModExt|args\l|build()\lpost_build()\l}"
    ]
  ]
  node [
    id 22
    name "22"
    label "{CLASSIFICATION|name\lvalue\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CLASSIFICATION|name\lvalue\l|}"
    ]
  ]
  node [
    id 23
    name "23"
    label "{CacheDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CacheDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 24
    name "24"
    label "{CachingDatasetFactory|cache_dir\ldatasets_dir\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CachingDatasetFactory|cache_dir\ldatasets_dir\l|}"
    ]
  ]
  node [
    id 25
    name "25"
    label "{CamVidDataset|class_groups_colors : dict\lcolor_to_label : dict\lsubsets : list\l|download()\lget_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CamVidDataset|class_groups_colors : dict\lcolor_to_label : dict\lsubsets : list\l|download()\lget_example()\l}"
    ]
  ]
  node [
    id 26
    name "26"
    label "{CarliniWagnerL2Attack|abort_early : bool\lbinary_search_steps : int\lconfidence : int\ldistance_fn\linitial_const : float\llearning_rate : float\lmax_iter : int\lnum_classes : NoneType\lrepeat\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CarliniWagnerL2Attack|abort_early : bool\lbinary_search_steps : int\lconfidence : int\ldistance_fn\linitial_const : float\llearning_rate : float\lmax_iter : int\lnum_classes : NoneType\lrepeat\l|}"
    ]
  ]
  node [
    id 27
    name "27"
    label "{CarliniWagnerLoss|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CarliniWagnerLoss|\l|forward()\l}"
    ]
  ]
  node [
    id 28
    name "28"
    label "{CheckpointManager|INFO_FILENAME : str\lLOG_FILENAME : str\lSTATE_FILENAME : str\ldir_path\lid\lsaved\l|load_last()\lremove_old_checkpoints()\lsave()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CheckpointManager|INFO_FILENAME : str\lLOG_FILENAME : str\lSTATE_FILENAME : str\ldir_path\lid\lsaved\l|load_last()\lremove_old_checkpoints()\lsave()\l}"
    ]
  ]
  node [
    id 29
    name "29"
    label "{Cifar100Dataset|subsets : list\lx\ly\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Cifar100Dataset|subsets : list\lx\ly\l|get_example()\l}"
    ]
  ]
  node [
    id 30
    name "30"
    label "{Cifar10Dataset|subsets : list\lx\ly : list\l|download()\lget_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Cifar10Dataset|subsets : list\lx\ly : list\l|download()\lget_example()\l}"
    ]
  ]
  node [
    id 31
    name "31"
    label "{CityscapesDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CityscapesDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 32
    name "32"
    label "{ClassificationHead|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ClassificationHead|\l|}"
    ]
  ]
  node [
    id 33
    name "33"
    label "{ClassificationMetrics|active : bool\lclass_count\lcm\llabels\l|compute()\lreset()\lupdate()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ClassificationMetrics|active : bool\lclass_count\lcm\llabels\l|compute()\lreset()\lupdate()\l}"
    ]
  ]
  node [
    id 34
    name "34"
    label "{ClassificationModel|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ClassificationModel|\l|}"
    ]
  ]
  node [
    id 35
    name "35"
    label "{ClassificationTrainer|\l|get_outputs()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ClassificationTrainer|\l|get_outputs()\l}"
    ]
  ]
  node [
    id 36
    name "36"
    label "{CollateDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{CollateDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 37
    name "37"
    label "{Concat|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Concat|\l|forward()\l}"
    ]
  ]
  node [
    id 38
    name "38"
    label "{Conv|orig\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Conv|orig\l|build()\l}"
    ]
  ]
  node [
    id 39
    name "39"
    label "{Conv2d|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Conv2d|\l|forward()\l}"
    ]
  ]
  node [
    id 40
    name "40"
    label "{ConvTranspose|orig : NoneType\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ConvTranspose|orig : NoneType\l|build()\l}"
    ]
  ]
  node [
    id 41
    name "41"
    label "{DEPTH_REGRESSION|name\lvalue\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DEPTH_REGRESSION|name\lvalue\l|}"
    ]
  ]
  node [
    id 42
    name "42"
    label "{DataLoader|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DataLoader|\l|}"
    ]
  ]
  node [
    id 43
    name "43"
    label "{Dataset|data : NoneType\lidentifier\linfo\lmodifiers : list\lname : str, NoneType\lsubset : NoneType\l|approx_example_sizeof()\lbatch()\lcache()\lcache_hdd()\lclear_hdd_cache()\lcollate()\ldownload()\ldownload_if_necessary()\lfilter()\lget_example()\ljoin()\lmap()\lpermute()\lrandom()\lrepeat()\lsample()\lsplit()\lzip()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Dataset|data : NoneType\lidentifier\linfo\lmodifiers : list\lname : str, NoneType\lsubset : NoneType\l|approx_example_sizeof()\lbatch()\lcache()\lcache_hdd()\lclear_hdd_cache()\lcollate()\ldownload()\ldownload_if_necessary()\lfilter()\lget_example()\ljoin()\lmap()\lpermute()\lrandom()\lrepeat()\lsample()\lsplit()\lzip()\l}"
    ]
  ]
  node [
    id 44
    name "44"
    label "{DatasetFactory|datasets_dir\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DatasetFactory|datasets_dir\l|}"
    ]
  ]
  node [
    id 45
    name "45"
    label "{DebbugableModExt|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DebbugableModExt|\l|forward()\l}"
    ]
  ]
  node [
    id 46
    name "46"
    label "{DenseBlock|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseBlock|\l|}"
    ]
  ]
  node [
    id 47
    name "47"
    label "{DenseNet|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseNet|\l|}"
    ]
  ]
  node [
    id 48
    name "48"
    label "{DenseNet|avgpool_size\lclassifier\lfeatures\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseNet|avgpool_size\lclassifier\lfeatures\l|forward()\l}"
    ]
  ]
  node [
    id 49
    name "49"
    label "{DenseNetBackbone|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseNetBackbone|\l|}"
    ]
  ]
  node [
    id 50
    name "50"
    label "{DenseNetCifarTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseNetCifarTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    ]
  ]
  node [
    id 51
    name "51"
    label "{DenseSequence|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseSequence|\l|}"
    ]
  ]
  node [
    id 52
    name "52"
    label "{DenseSpatialPyramidPooling|grids : tuple\lspp\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseSpatialPyramidPooling|grids : tuple\lspp\l|forward()\l}"
    ]
  ]
  node [
    id 53
    name "53"
    label "{DenseTransition|args\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseTransition|args\l|build()\l}"
    ]
  ]
  node [
    id 54
    name "54"
    label "{DenseUnit|block\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DenseUnit|block\l|forward()\l}"
    ]
  ]
  node [
    id 55
    name "55"
    label "{DescribableTexturesDataset|data\lsubsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DescribableTexturesDataset|data\lsubsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 56
    name "56"
    label "{DiscriminativeModel|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{DiscriminativeModel|\l|}"
    ]
  ]
  node [
    id 57
    name "57"
    label "{Dropout|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Dropout|\l|forward()\l}"
    ]
  ]
  node [
    id 58
    name "58"
    label "{Dropout2d|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Dropout2d|\l|forward()\l}"
    ]
  ]
  node [
    id 59
    name "59"
    label "{Engine|completed\lepoch_completed\lepoch_started\literation_completed\literation_started\lshould_terminate : bool\lshould_terminate_single_epoch : bool\lstarted\lstate\l|load_state_dict()\lrun()\lstate_dict()\lterminate()\lterminate_epoch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Engine|completed\lepoch_completed\lepoch_started\literation_completed\literation_started\lshould_terminate : bool\lshould_terminate_single_epoch : bool\lstarted\lstate\l|load_state_dict()\lrun()\lstate_dict()\lterminate()\lterminate_epoch()\l}"
    ]
  ]
  node [
    id 60
    name "60"
    label "{Evaluator|data_loader_f\leval_step\leval_step\levaluation\lget_outputs\lget_outputs\lloss\lmetrics\lmodel\lprepare_batch\lprepare_batch\l|eval()\lget_metric_values()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Evaluator|data_loader_f\leval_step\leval_step\levaluation\lget_outputs\lget_outputs\lloss\lmetrics\lmodel\lprepare_batch\lprepare_batch\l|eval()\lget_metric_values()\l}"
    ]
  ]
  node [
    id 61
    name "61"
    label "{Evaluator|config\levaluation\lmetrics : dict\l|attach_metric()\leval()\leval_batch()\lget_metric_values()\lget_outputs()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Evaluator|config\levaluation\lmetrics : dict\l|attach_metric()\leval()\leval_batch()\lget_metric_values()\lget_outputs()\l}"
    ]
  ]
  node [
    id 62
    name "62"
    label "{EvaluatorConfig|data_loader_f\lloss\lloss\lloss_f\lmodel\lprepare_batch\lprepare_batch\l|check_all_initialized()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{EvaluatorConfig|data_loader_f\lloss\lloss\lloss_f\lmodel\lprepare_batch\lprepare_batch\l|check_all_initialized()\l}"
    ]
  ]
  node [
    id 63
    name "63"
    label "{Event|handlers : list\l|add_handler()\lhandler()\lremove_handler()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Event|handlers : list\l|add_handler()\lhandler()\lremove_handler()\l}"
    ]
  ]
  node [
    id 64
    name "64"
    label "{ExtendedInterfaceModExt|device\l|add_module()\ladd_modules()\lload_state_dict()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ExtendedInterfaceModExt|device\l|add_module()\ladd_modules()\lload_state_dict()\l}"
    ]
  ]
  node [
    id 65
    name "65"
    label "{FCNEncoder|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{FCNEncoder|\l|}"
    ]
  ]
  node [
    id 66
    name "66"
    label "{FDenseBlock|block_ends\lblock_start_columns\llength\lsum\lwidth\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{FDenseBlock|block_ends\lblock_start_columns\llength\lsum\lwidth\l|forward()\l}"
    ]
  ]
  node [
    id 67
    name "67"
    label "{FDenseNetBackbone|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{FDenseNetBackbone|\l|}"
    ]
  ]
  node [
    id 68
    name "68"
    label "{FDenseSequence|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{FDenseSequence|\l|}"
    ]
  ]
  node [
    id 69
    name "69"
    label "{Field|compare\ldefault\ldefault_factory\lhash\linit\lmetadata\lname : NoneType\lrepr\ltype : NoneType\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Field|compare\ldefault\ldefault_factory\lhash\linit\lmetadata\lname : NoneType\lrepr\ltype : NoneType\l|}"
    ]
  ]
  node [
    id 70
    name "70"
    label "{Func|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Func|\l|forward()\l}"
    ]
  ]
  node [
    id 71
    name "71"
    label "{FuncMetric|func\lname : str\l|compute()\lreset()\lupdate()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{FuncMetric|func\lname : str\l|compute()\lreset()\lupdate()\l}"
    ]
  ]
  node [
    id 72
    name "72"
    label "{GAN|discriminator\lgenerator\lz_rand\lz_shape\l|sample_z()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{GAN|discriminator\lgenerator\lz_rand\lz_shape\l|sample_z()\l}"
    ]
  ]
  node [
    id 73
    name "73"
    label "{GANTrainer|\l|train_step()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{GANTrainer|\l|train_step()\l}"
    ]
  ]
  node [
    id 74
    name "74"
    label "{GANTrainer|\l|train_batch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{GANTrainer|\l|train_batch()\l}"
    ]
  ]
  node [
    id 75
    name "75"
    label "{GANTrainerConfig|loss\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{GANTrainerConfig|loss\l|}"
    ]
  ]
  node [
    id 76
    name "76"
    label "{GradientSignAttack|eps : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{GradientSignAttack|eps : float\l|}"
    ]
  ]
  node [
    id 77
    name "77"
    label "{HBlobsDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{HBlobsDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 78
    name "78"
    label "{HDDAndRAMCacheDataset|cache_dir\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{HDDAndRAMCacheDataset|cache_dir\l|}"
    ]
  ]
  node [
    id 79
    name "79"
    label "{HDDCacheDataset|cache_dir\lkeys : list\lseparate_fields : bool\l|delete_cache()\lget_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{HDDCacheDataset|cache_dir\lkeys : list\lseparate_fields : bool\l|delete_cache()\lget_example()\l}"
    ]
  ]
  node [
    id 80
    name "80"
    label "{ICCV09Dataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ICCV09Dataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 81
    name "81"
    label "{INaturalist2018Dataset|categories : str\lsubsets : tuple\lurl : str\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{INaturalist2018Dataset|categories : str\lsubsets : tuple\lurl : str\l|get_example()\l}"
    ]
  ]
  node [
    id 82
    name "82"
    label "{ISUNDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ISUNDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 83
    name "83"
    label "{Identity|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Identity|\l|forward()\l}"
    ]
  ]
  node [
    id 84
    name "84"
    label "{Identity|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Identity|\l|forward()\l}"
    ]
  ]
  node [
    id 85
    name "85"
    label "{ImageFolder|imgs : list\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ImageFolder|imgs : list\l|}"
    ]
  ]
  node [
    id 86
    name "86"
    label "{ImageTransformer|layout : str\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ImageTransformer|layout : str\l|}"
    ]
  ]
  node [
    id 87
    name "87"
    label "{IntermediateOutputsModuleWrapper|handles : NoneType\lmodule\loutputs : NoneType\l|forward()\lpost_build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{IntermediateOutputsModuleWrapper|handles : NoneType\lmodule\loutputs : NoneType\l|forward()\lpost_build()\l}"
    ]
  ]
  node [
    id 88
    name "88"
    label "{LSUNDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{LSUNDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 89
    name "89"
    label "{Ladder|upsample_blends\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Ladder|upsample_blends\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 90
    name "90"
    label "{LadderDenseNetTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{LadderDenseNetTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    ]
  ]
  node [
    id 91
    name "91"
    label "{LadderUpsampleBlend|blend\ljoin\lproject\lupsample\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{LadderUpsampleBlend|blend\ljoin\lproject\lupsample\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 92
    name "92"
    label "{LazyImageStatisticsComputer|cache_dir\lcache_path\ldataset\linitialized\lsingle_channel\lstats\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{LazyImageStatisticsComputer|cache_dir\lcache_path\ldataset\linitialized\lsingle_channel\lstats\l|}"
    ]
  ]
  node [
    id 93
    name "93"
    label "{Linear|bias\lin_features\lout_features\lweight\l|extra_repr()\lforward()\lreset_parameters()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Linear|bias\lin_features\lout_features\lweight\l|extra_repr()\lforward()\lreset_parameters()\l}"
    ]
  ]
  node [
    id 94
    name "94"
    label "{Linear|orig\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Linear|orig\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 95
    name "95"
    label "{Logger|lines : list\lprint_verbosity_threshold : int\lverbosities : list\l|load_state_dict()\llog()\lprint_all()\lstate_dict()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Logger|lines : list\lprint_verbosity_threshold : int\lverbosities : list\l|load_state_dict()\llog()\lprint_all()\lstate_dict()\l}"
    ]
  ]
  node [
    id 96
    name "96"
    label "{Logger|disabled : bool\lfatal\lhandlers : list\llevel : int\lmanager\lname\lparent : NoneType\lpropagate : bool\l|addHandler()\lcallHandlers()\lcritical()\ldebug()\lerror()\lexception()\lfindCaller()\lgetChild()\lgetEffectiveLevel()\lhandle()\lhasHandlers()\linfo()\lisEnabledFor()\llog()\lmakeRecord()\lremoveHandler()\lsetLevel()\lwarn()\lwarning()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Logger|disabled : bool\lfatal\lhandlers : list\llevel : int\lmanager\lname\lparent : NoneType\lpropagate : bool\l|addHandler()\lcallHandlers()\lcritical()\ldebug()\lerror()\lexception()\lfindCaller()\lgetChild()\lgetEffectiveLevel()\lhandle()\lhasHandlers()\linfo()\lisEnabledFor()\llog()\lmakeRecord()\lremoveHandler()\lsetLevel()\lwarn()\lwarning()\l}"
    ]
  ]
  node [
    id 97
    name "97"
    label "{MDenseBlock|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MDenseBlock|\l|}"
    ]
  ]
  node [
    id 98
    name "98"
    label "{MDenseNetBackbone|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MDenseNetBackbone|\l|}"
    ]
  ]
  node [
    id 99
    name "99"
    label "{MDenseSequence|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MDenseSequence|\l|}"
    ]
  ]
  node [
    id 100
    name "100"
    label "{MDenseTransition|args\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MDenseTransition|args\l|build()\l}"
    ]
  ]
  node [
    id 101
    name "101"
    label "{MDenseUnit|block_end\lblock_starts\lsum\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MDenseUnit|block_end\lblock_starts\lsum\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 102
    name "102"
    label "{MNISTDataset|subsets : list\lx\ly\l|download()\lget_example()\lload_array()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MNISTDataset|subsets : list\lx\ly\l|download()\lget_example()\lload_array()\l}"
    ]
  ]
  node [
    id 103
    name "103"
    label "{Manager|disable : int\lemittedNoHandlerWarning : bool\llogRecordFactory : NoneType\lloggerClass : NoneType\lloggerDict : dict\lroot\l|getLogger()\lsetLogRecordFactory()\lsetLoggerClass()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Manager|disable : int\lemittedNoHandlerWarning : bool\llogRecordFactory : NoneType\lloggerClass : NoneType\lloggerDict : dict\lroot\l|getLogger()\lsetLogRecordFactory()\lsetLoggerClass()\l}"
    ]
  ]
  node [
    id 104
    name "104"
    label "{MapDataset|func\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MapDataset|func\l|get_example()\l}"
    ]
  ]
  node [
    id 105
    name "105"
    label "{MaxPool|orig\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MaxPool|orig\l|build()\l}"
    ]
  ]
  node [
    id 106
    name "106"
    label "{MaxPool2d|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{MaxPool2d|\l|forward()\l}"
    ]
  ]
  node [
    id 107
    name "107"
    label "{Missing|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Missing|\l|}"
    ]
  ]
  node [
    id 108
    name "108"
    label "{Model|\l|initialize()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Model|\l|initialize()\l}"
    ]
  ]
  node [
    id 109
    name "109"
    label "{Module|args\ldevice\l|build()\lload_state_dict()\lpost_build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Module|args\ldevice\l|build()\lload_state_dict()\lpost_build()\l}"
    ]
  ]
  node [
    id 110
    name "110"
    label "{ModuleList|\l|append()\lextend()\linsert()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ModuleList|\l|append()\lextend()\linsert()\l}"
    ]
  ]
  node [
    id 111
    name "111"
    label "{NameDict|in_features\lpadding\l|as_dict()\litems()\lkeys()\lvalues()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{NameDict|in_features\lpadding\l|as_dict()\litems()\lkeys()\lvalues()\l}"
    ]
  ]
  node [
    id 112
    name "112"
    label "{NameDict|in_features\lpadding\l|as_dict()\litems()\lkeys()\lvalues()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{NameDict|in_features\lpadding\l|as_dict()\litems()\lkeys()\lvalues()\l}"
    ]
  ]
  node [
    id 113
    name "113"
    label "{NumPyImageTransformer|\l|center_crop()\lhflip()\lto_numpy()\lto_pil()\lto_torch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{NumPyImageTransformer|\l|center_crop()\lhflip()\lto_numpy()\lto_pil()\lto_torch()\l}"
    ]
  ]
  node [
    id 114
    name "114"
    label "{OTHER|name\lvalue\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{OTHER|name\lvalue\l|}"
    ]
  ]
  node [
    id 115
    name "115"
    label "{OrderedDict|index\l|move_to_end()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{OrderedDict|index\l|move_to_end()\l}"
    ]
  ]
  node [
    id 116
    name "116"
    label "{PGDAttack|eps : float\liter_count : int\lp\lrand_init : bool\lstep_size : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{PGDAttack|eps : float\liter_count : int\lp\lrand_init : bool\lstep_size : float\l|}"
    ]
  ]
  node [
    id 117
    name "117"
    label "{PILImageTransformer|\l|center_crop()\lhflip()\lpad()\lrand_crop()\lrand_hflip()\lto_numpy()\lto_pil()\lto_torch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{PILImageTransformer|\l|center_crop()\lhflip()\lpad()\lrand_crop()\lrand_hflip()\lto_numpy()\lto_pil()\lto_torch()\l}"
    ]
  ]
  node [
    id 118
    name "118"
    label "{Parallel|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Parallel|\l|forward()\l}"
    ]
  ]
  node [
    id 119
    name "119"
    label "{Parameter|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Parameter|\l|}"
    ]
  ]
  node [
    id 120
    name "120"
    label "{PartedDataset|part_to_ds : dict\ltop_level_parts\l|items()\lkeys()\ltop_level_items()\lwith_transform()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{PartedDataset|part_to_ds : dict\ltop_level_parts\l|items()\lkeys()\ltop_level_items()\lwith_transform()\l}"
    ]
  ]
  node [
    id 121
    name "121"
    label "{Path|\l|absolute()\lchmod()\lcwd()\lexists()\lexpanduser()\lglob()\lgroup()\lhome()\lis_block_device()\lis_char_device()\lis_dir()\lis_fifo()\lis_file()\lis_mount()\lis_socket()\lis_symlink()\literdir()\llchmod()\llstat()\lmkdir()\lopen()\lowner()\lread_bytes()\lread_text()\lrename()\lreplace()\lresolve()\lrglob()\lrmdir()\lsamefile()\lstat()\lsymlink_to()\ltouch()\lunlink()\lwrite_bytes()\lwrite_text()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Path|\l|absolute()\lchmod()\lcwd()\lexists()\lexpanduser()\lglob()\lgroup()\lhome()\lis_block_device()\lis_char_device()\lis_dir()\lis_fifo()\lis_file()\lis_mount()\lis_socket()\lis_symlink()\literdir()\llchmod()\llstat()\lmkdir()\lopen()\lowner()\lread_bytes()\lread_text()\lrename()\lreplace()\lresolve()\lrglob()\lrmdir()\lsamefile()\lstat()\lsymlink_to()\ltouch()\lunlink()\lwrite_bytes()\lwrite_text()\l}"
    ]
  ]
  node [
    id 122
    name "122"
    label "{PostactBlock|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{PostactBlock|\l|}"
    ]
  ]
  node [
    id 123
    name "123"
    label "{PreactBlock|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{PreactBlock|\l|}"
    ]
  ]
  node [
    id 124
    name "124"
    label "{Problem|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Problem|\l|}"
    ]
  ]
  node [
    id 125
    name "125"
    label "{RademacherNoiseDataset|subsets : list\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RademacherNoiseDataset|subsets : list\l|}"
    ]
  ]
  node [
    id 126
    name "126"
    label "{RandomDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RandomDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 127
    name "127"
    label "{ReLU|\l|extra_repr()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ReLU|\l|extra_repr()\l}"
    ]
  ]
  node [
    id 128
    name "128"
    label "{Record|\l|evaluate()\lis_evaluated()\litems()\ljoin()\lkeys()\lvalues()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Record|\l|evaluate()\lis_evaluated()\litems()\ljoin()\lkeys()\lvalues()\l}"
    ]
  ]
  node [
    id 129
    name "129"
    label "{Reduce|func\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Reduce|func\l|forward()\l}"
    ]
  ]
  node [
    id 130
    name "130"
    label "{RegressionHead|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RegressionHead|\l|}"
    ]
  ]
  node [
    id 131
    name "131"
    label "{RepeatDataset|number_of_repeats\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RepeatDataset|number_of_repeats\l|get_example()\l}"
    ]
  ]
  node [
    id 132
    name "132"
    label "{ResNet|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNet|\l|}"
    ]
  ]
  node [
    id 133
    name "133"
    label "{ResNet|avgpool\lconv1\lfc\linplanes : int\llayer1\llayer2\llayer3\llayer4\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNet|avgpool\lconv1\lfc\linplanes : int\llayer1\llayer2\llayer3\llayer4\l|forward()\l}"
    ]
  ]
  node [
    id 134
    name "134"
    label "{ResNet18|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNet18|\l|}"
    ]
  ]
  node [
    id 135
    name "135"
    label "{ResNetCifarTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNetCifarTrainerConfig|batch_size : int\lepoch_count : int\llr_scheduler_f\loptimizer_f\lweight_decay : float\l|}"
    ]
  ]
  node [
    id 136
    name "136"
    label "{ResNetV2Backbone|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNetV2Backbone|\l|}"
    ]
  ]
  node [
    id 137
    name "137"
    label "{ResNetV2Groups|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNetV2Groups|\l|}"
    ]
  ]
  node [
    id 138
    name "138"
    label "{ResNetV2Unit|block\lpreact\lshortcut : NoneType\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ResNetV2Unit|block\lpreact\lshortcut : NoneType\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 139
    name "139"
    label "{Resize|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Resize|\l|forward()\l}"
    ]
  ]
  node [
    id 140
    name "140"
    label "{RootBlock|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RootBlock|\l|}"
    ]
  ]
  node [
    id 141
    name "141"
    label "{RootLogger|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{RootLogger|\l|}"
    ]
  ]
  node [
    id 142
    name "142"
    label "{SEMANTIC_SEGMENTATION|name\lvalue\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SEMANTIC_SEGMENTATION|name\lvalue\l|}"
    ]
  ]
  node [
    id 143
    name "143"
    label "{SVHNDataset|subsets : list\lx\ly\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SVHNDataset|subsets : list\lx\ly\l|get_example()\l}"
    ]
  ]
  node [
    id 144
    name "144"
    label "{SampleDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SampleDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 145
    name "145"
    label "{ScalableLambdaLR|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ScalableLambdaLR|\l|}"
    ]
  ]
  node [
    id 146
    name "146"
    label "{ScalableMultiStepLR|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ScalableMultiStepLR|\l|}"
    ]
  ]
  node [
    id 147
    name "147"
    label "{ScopedModuleExtension|scope\lscopei\l|add_module()\lget_parents()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ScopedModuleExtension|scope\lscopei\l|add_module()\lget_parents()\l}"
    ]
  ]
  node [
    id 148
    name "148"
    label "{SegmentationHead|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SegmentationHead|\l|}"
    ]
  ]
  node [
    id 149
    name "149"
    label "{SeqModel|initialize\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SeqModel|initialize\l|}"
    ]
  ]
  node [
    id 150
    name "150"
    label "{Sequential|\l|index()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Sequential|\l|index()\l}"
    ]
  ]
  node [
    id 151
    name "151"
    label "{Sequential|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Sequential|\l|forward()\l}"
    ]
  ]
  node [
    id 152
    name "152"
    label "{SimpleEncoder|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SimpleEncoder|\l|}"
    ]
  ]
  node [
    id 153
    name "153"
    label "{SlugUnit|blocks : NoneType\l|build()\lforward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SlugUnit|blocks : NoneType\l|build()\lforward()\l}"
    ]
  ]
  node [
    id 154
    name "154"
    label "{SmallImageClassifier|conv1\lconv2\lfc1\lfc2\lfc3\lpool\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SmallImageClassifier|conv1\lconv2\lfc1\lfc2\lfc3\lpool\l|forward()\l}"
    ]
  ]
  node [
    id 155
    name "155"
    label "{SmallImageClassifierTrainerConfig|batch_size : int\lepoch_count : int\loptimizer_f\lweight_decay : int\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SmallImageClassifierTrainerConfig|batch_size : int\lepoch_count : int\loptimizer_f\lweight_decay : int\l|}"
    ]
  ]
  node [
    id 156
    name "156"
    label "{State|batch : NoneType\lbatch_count : NoneType\lepoch : int\literation : int\loutput : NoneType\l|reset()\lupdate()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{State|batch : NoneType\lbatch_count : NoneType\lepoch : int\literation : int\loutput : NoneType\l|reset()\lupdate()\l}"
    ]
  ]
  node [
    id 157
    name "157"
    label "{StochasticModExt|\l|eval()\lis_stochastic()\lsample()\lsample1()\lstochastic_eval()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{StochasticModExt|\l|eval()\lis_stochastic()\lsample()\lsample1()\lstochastic_eval()\l}"
    ]
  ]
  node [
    id 158
    name "158"
    label "{StochasticModule|\l|deterministic_forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{StochasticModule|\l|deterministic_forward()\l}"
    ]
  ]
  node [
    id 159
    name "159"
    label "{SubDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SubDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 160
    name "160"
    label "{SubrangeDataset|start\lstep\lstop\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SubrangeDataset|start\lstep\lstop\l|get_example()\l}"
    ]
  ]
  node [
    id 161
    name "161"
    label "{Sum|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Sum|\l|forward()\l}"
    ]
  ]
  node [
    id 162
    name "162"
    label "{SupervisedTrainer|\l|eval_batch()\ltrain_batch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SupervisedTrainer|\l|eval_batch()\ltrain_batch()\l}"
    ]
  ]
  node [
    id 163
    name "163"
    label "{Synchronized|value\lvalue : bool\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Synchronized|value\lvalue : bool\l|}"
    ]
  ]
  node [
    id 164
    name "164"
    label "{SynchronizedArray|value : bool\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SynchronizedArray|value : bool\l|}"
    ]
  ]
  node [
    id 165
    name "165"
    label "{SynchronizedString|raw\lvalue\lvalue : bool\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{SynchronizedString|raw\lvalue\lvalue : bool\l|}"
    ]
  ]
  node [
    id 166
    name "166"
    label "{TCSegmentationHead|act_f\lclass_count\lconvt_f\lnorm_f\lshape\l|build()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TCSegmentationHead|act_f\lclass_count\lconvt_f\lnorm_f\lshape\l|build()\l}"
    ]
  ]
  node [
    id 167
    name "167"
    label "{TinyImageNetDataset|name\lsubsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TinyImageNetDataset|name\lsubsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 168
    name "168"
    label "{TinyImagesDataset|cifar_idxs : list, tuple\lexclude_cifar : bool\lin_cifar\lload_image\lsubsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TinyImagesDataset|cifar_idxs : list, tuple\lexclude_cifar : bool\lin_cifar\lload_image\lsubsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 169
    name "169"
    label "{TorchImageTransformer|\l|destandardize()\lresize()\lscale()\lstandardize()\lto_float32()\lto_numpy()\lto_pil()\lto_torch()\lto_uint8()\ltranspose_to_chw()\ltranspose_to_hwc()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TorchImageTransformer|\l|destandardize()\lresize()\lscale()\lstandardize()\lto_float32()\lto_numpy()\lto_pil()\lto_torch()\lto_uint8()\ltranspose_to_chw()\ltranspose_to_hwc()\l}"
    ]
  ]
  node [
    id 170
    name "170"
    label "{Trainer|batch_size : int\lepoch_count\llr_scheduler\llr_scheduler\llr_scheduler_f\loptimizer\loptimizer\loptimizer_f\lstate_attrs : tuple\ltrain_step\ltrain_step\ltraining\lweight_decay\l|load_state_dict()\lstate_dict()\ltrain()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Trainer|batch_size : int\lepoch_count\llr_scheduler\llr_scheduler\llr_scheduler_f\loptimizer\loptimizer\loptimizer_f\lstate_attrs : tuple\ltrain_step\ltrain_step\ltraining\lweight_decay\l|load_state_dict()\lstate_dict()\ltrain()\l}"
    ]
  ]
  node [
    id 171
    name "171"
    label "{Trainer|state_attrs : tuple\ltraining\l|load_state_dict()\lstate_dict()\ltrain()\ltrain_batch()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Trainer|state_attrs : tuple\ltraining\l|load_state_dict()\lstate_dict()\ltrain()\ltrain_batch()\l}"
    ]
  ]
  node [
    id 172
    name "172"
    label "{TrainerConfig|batch_size : int\lepoch_count\llr_scheduler\llr_scheduler\llr_scheduler_f\loptimizer\loptimizer\loptimizer_f\lweight_decay\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TrainerConfig|batch_size : int\lepoch_count\llr_scheduler\llr_scheduler\llr_scheduler_f\loptimizer\loptimizer\loptimizer_f\lweight_decay\l|}"
    ]
  ]
  node [
    id 173
    name "173"
    label "{TrainingExperiment|cpman\ldata\llogger\lmodel\ltrainer\l|create()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{TrainingExperiment|cpman\ldata\llogger\lmodel\ltrainer\l|create()\l}"
    ]
  ]
  node [
    id 174
    name "174"
    label "{Transformer|\l|transform()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{Transformer|\l|transform()\l}"
    ]
  ]
  node [
    id 175
    name "175"
    label "{VATLoss|eps : float\liter_count : int\lxi : float\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{VATLoss|eps : float\liter_count : int\lxi : float\l|forward()\l}"
    ]
  ]
  node [
    id 176
    name "176"
    label "{VGGBackbone|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{VGGBackbone|\l|}"
    ]
  ]
  node [
    id 177
    name "177"
    label "{VGGClassifier|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{VGGClassifier|\l|}"
    ]
  ]
  node [
    id 178
    name "178"
    label "{VOC2012SegmentationDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{VOC2012SegmentationDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 179
    name "179"
    label "{WRNCifarTrainerConfig|weight_decay : float\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{WRNCifarTrainerConfig|weight_decay : float\l|}"
    ]
  ]
  node [
    id 180
    name "180"
    label "{WhiteNoiseDataset|subsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{WhiteNoiseDataset|subsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 181
    name "181"
    label "{WideResNet|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{WideResNet|\l|}"
    ]
  ]
  node [
    id 182
    name "182"
    label "{WildDashDataset|splits : dict\lsubsets : list\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{WildDashDataset|splits : dict\lsubsets : list\l|get_example()\l}"
    ]
  ]
  node [
    id 183
    name "183"
    label "{WrappedModule|orig : NoneType\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{WrappedModule|orig : NoneType\l|forward()\l}"
    ]
  ]
  node [
    id 184
    name "184"
    label "{YingzhenLiEncoder|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{YingzhenLiEncoder|\l|}"
    ]
  ]
  node [
    id 185
    name "185"
    label "{ZipDataset|\l|get_example()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{ZipDataset|\l|get_example()\l}"
    ]
  ]
  node [
    id 186
    name "186"
    label "{_Bottleneck|bn1\lbn2\lbn3\lconv1\lconv2\lconv3\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_Bottleneck|bn1\lbn2\lbn3\lconv1\lconv2\lconv3\ldownsample : NoneType\lexpansion : int\lrelu\lstride : int\l|forward()\l}"
    ]
  ]
  node [
    id 187
    name "187"
    label "{_DenseBlock|\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_DenseBlock|\l|forward()\l}"
    ]
  ]
  node [
    id 188
    name "188"
    label "{_DenseLayer|drop_rate\lefficient : bool\l|forward()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_DenseLayer|drop_rate\lefficient : bool\l|forward()\l}"
    ]
  ]
  node [
    id 189
    name "189"
    label "{_DropoutNd|inplace : bool\lp : float\l|extra_repr()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_DropoutNd|inplace : bool\lp : float\l|extra_repr()\l}"
    ]
  ]
  node [
    id 190
    name "190"
    label "{_FIELD_BASE|name\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_FIELD_BASE|name\l|}"
    ]
  ]
  node [
    id 191
    name "191"
    label "{_LazyField|get\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_LazyField|get\l|}"
    ]
  ]
  node [
    id 192
    name "192"
    label "{_NormalAccessor|chmod\llchmod\llistdir\llstat\lmkdir\lopen\lrename\lreplace\lrmdir\lscandir\lstat\lsymlink\lunlink\lutime\l|readlink()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_NormalAccessor|chmod\llchmod\llistdir\llstat\lmkdir\lopen\lrename\lreplace\lrmdir\lscandir\lstat\lsymlink\lunlink\lutime\l|readlink()\l}"
    ]
  ]
  node [
    id 193
    name "193"
    label "{_PartSplit|ratio\lsubparts\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_PartSplit|ratio\lsubparts\l|}"
    ]
  ]
  node [
    id 194
    name "194"
    label "{_Transition|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_Transition|\l|}"
    ]
  ]
  node [
    id 195
    name "195"
    label "{_empty|\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{_empty|\l|}"
    ]
  ]
  node [
    id 196
    name "196"
    label "{defaultdict|default_factory : NoneType\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{defaultdict|default_factory : NoneType\l|}"
    ]
  ]
  node [
    id 197
    name "197"
    label "{dtype|alignment : NoneType\lbase : NoneType\lbyteorder : NoneType\lchar : NoneType\ldescr : NoneType\lfields : NoneType\lflags : NoneType\lhasobject : NoneType\lisalignedstruct : NoneType\lisbuiltin : NoneType\lisnative : NoneType\litemsize : NoneType\lkind : NoneType\lmetadata : NoneType\lname : NoneType\lnames : NoneType, tuple, list\lnum : NoneType\lshape : NoneType\lstr : NoneType\lsubdtype : NoneType\ltype : NoneType\l|newbyteorder()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{dtype|alignment : NoneType\lbase : NoneType\lbyteorder : NoneType\lchar : NoneType\ldescr : NoneType\lfields : NoneType\lflags : NoneType\lhasobject : NoneType\lisalignedstruct : NoneType\lisbuiltin : NoneType\lisnative : NoneType\litemsize : NoneType\lkind : NoneType\lmetadata : NoneType\lname : NoneType\lnames : NoneType, tuple, list\lnum : NoneType\lshape : NoneType\lstr : NoneType\lsubdtype : NoneType\ltype : NoneType\l|newbyteorder()\l}"
    ]
  ]
  node [
    id 198
    name "198"
    label "{partial|args : tuple\lfunc\lkeywords : dict\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{partial|args : tuple\lfunc\lkeywords : dict\l|}"
    ]
  ]
  node [
    id 199
    name "199"
    label "{partial|args : tuple\lfunc\lkeywords : dict\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{partial|args : tuple\lfunc\lkeywords : dict\l|}"
    ]
  ]
  node [
    id 200
    name "200"
    label "{partialmethod|args : tuple\lfunc\lkeywords : dict\l|}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{partialmethod|args : tuple\lfunc\lkeywords : dict\l|}"
    ]
  ]
  node [
    id 201
    name "201"
    label "{recarray|dtype : NoneType\lfreq : NoneType\lshape : tuple\l|field()\l}"
    graphics [
      type "record"
    ]
    LabelGraphics [
      text "{recarray|dtype : NoneType\lfreq : NoneType\lshape : tuple\l|field()\l}"
    ]
  ]
  edge [
    id 1
    source 0
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 126
    source 1
    target 5
    label "discriminator"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "discriminator"
      fontColor "green"
    ]
  ]
  edge [
    id 2
    source 1
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 3
    source 2
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 127
    source 4
    target 133
    label "avgpool"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "avgpool"
      fontColor "green"
    ]
  ]
  edge [
    id 4
    source 5
    target 10
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 5
    source 6
    target 162
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 6
    source 7
    target 170
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 7
    source 8
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 8
    source 10
    target 108
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 9
    source 11
    target 171
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 10
    source 12
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 128
    source 13
    target 14
    label "avgpool"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "avgpool"
      fontColor "green"
    ]
  ]
  edge [
    id 11
    source 16
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 12
    source 17
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 129
    source 18
    target 14
    label "bn1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn1"
      fontColor "green"
    ]
  ]
  edge [
    id 130
    source 18
    target 15
    label "bn1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn1"
      fontColor "green"
    ]
  ]
  edge [
    id 131
    source 18
    target 15
    label "bn2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn2"
      fontColor "green"
    ]
  ]
  edge [
    id 132
    source 18
    target 19
    label "bn1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn1"
      fontColor "green"
    ]
  ]
  edge [
    id 133
    source 18
    target 19
    label "bn2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn2"
      fontColor "green"
    ]
  ]
  edge [
    id 134
    source 18
    target 19
    label "bn3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn3"
      fontColor "green"
    ]
  ]
  edge [
    id 135
    source 18
    target 186
    label "bn1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn1"
      fontColor "green"
    ]
  ]
  edge [
    id 136
    source 18
    target 186
    label "bn2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn2"
      fontColor "green"
    ]
  ]
  edge [
    id 137
    source 18
    target 186
    label "bn3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bn3"
      fontColor "green"
    ]
  ]
  edge [
    id 13
    source 23
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 14
    source 24
    target 44
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 15
    source 25
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 16
    source 26
    target 9
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 17
    source 29
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 18
    source 30
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 19
    source 31
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 20
    source 32
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 21
    source 33
    target 3
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 22
    source 34
    target 56
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 23
    source 35
    target 162
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 24
    source 36
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 25
    source 37
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 138
    source 38
    target 138
    label "shortcut"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "shortcut"
      fontColor "green"
    ]
  ]
  edge [
    id 139
    source 38
    target 138
    label "shortcut"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "shortcut"
      fontColor "green"
    ]
  ]
  edge [
    id 26
    source 38
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 140
    source 39
    target 15
    label "conv1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv1"
      fontColor "green"
    ]
  ]
  edge [
    id 141
    source 39
    target 15
    label "conv2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv2"
      fontColor "green"
    ]
  ]
  edge [
    id 142
    source 39
    target 19
    label "conv1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv1"
      fontColor "green"
    ]
  ]
  edge [
    id 143
    source 39
    target 19
    label "conv2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv2"
      fontColor "green"
    ]
  ]
  edge [
    id 144
    source 39
    target 19
    label "conv3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv3"
      fontColor "green"
    ]
  ]
  edge [
    id 145
    source 39
    target 133
    label "conv1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv1"
      fontColor "green"
    ]
  ]
  edge [
    id 146
    source 39
    target 154
    label "conv1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv1"
      fontColor "green"
    ]
  ]
  edge [
    id 147
    source 39
    target 154
    label "conv2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv2"
      fontColor "green"
    ]
  ]
  edge [
    id 148
    source 39
    target 186
    label "conv1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv1"
      fontColor "green"
    ]
  ]
  edge [
    id 149
    source 39
    target 186
    label "conv2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv2"
      fontColor "green"
    ]
  ]
  edge [
    id 150
    source 39
    target 186
    label "conv3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "conv3"
      fontColor "green"
    ]
  ]
  edge [
    id 27
    source 40
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 28
    source 46
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 29
    source 47
    target 34
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 30
    source 49
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 31
    source 50
    target 172
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 32
    source 51
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 33
    source 53
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 34
    source 54
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 35
    source 55
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 36
    source 56
    target 149
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 37
    source 57
    target 189
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 38
    source 58
    target 189
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 151
    source 59
    target 60
    label "evaluation"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "evaluation"
      fontColor "green"
    ]
  ]
  edge [
    id 152
    source 59
    target 61
    label "evaluation"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "evaluation"
      fontColor "green"
    ]
  ]
  edge [
    id 153
    source 59
    target 170
    label "training"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "training"
      fontColor "green"
    ]
  ]
  edge [
    id 154
    source 59
    target 171
    label "training"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "training"
      fontColor "green"
    ]
  ]
  edge [
    id 155
    source 63
    target 59
    label "started"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "started"
      fontColor "green"
    ]
  ]
  edge [
    id 156
    source 63
    target 59
    label "completed"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "completed"
      fontColor "green"
    ]
  ]
  edge [
    id 157
    source 63
    target 59
    label "epoch_started"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "epoch_started"
      fontColor "green"
    ]
  ]
  edge [
    id 158
    source 63
    target 59
    label "epoch_completed"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "epoch_completed"
      fontColor "green"
    ]
  ]
  edge [
    id 159
    source 63
    target 59
    label "iteration_started"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "iteration_started"
      fontColor "green"
    ]
  ]
  edge [
    id 160
    source 63
    target 59
    label "iteration_completed"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "iteration_completed"
      fontColor "green"
    ]
  ]
  edge [
    id 39
    source 65
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 40
    source 66
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 41
    source 67
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 42
    source 68
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 161
    source 69
    target 7
    label "attack"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "attack"
      fontColor "green"
    ]
  ]
  edge [
    id 162
    source 69
    target 60
    label "metrics"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "metrics"
      fontColor "green"
    ]
  ]
  edge [
    id 163
    source 69
    target 62
    label "loss"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "loss"
      fontColor "green"
    ]
  ]
  edge [
    id 164
    source 69
    target 75
    label "loss"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "loss"
      fontColor "green"
    ]
  ]
  edge [
    id 165
    source 69
    target 170
    label "optimizer"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer"
      fontColor "green"
    ]
  ]
  edge [
    id 166
    source 69
    target 170
    label "lr_scheduler"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler"
      fontColor "green"
    ]
  ]
  edge [
    id 167
    source 69
    target 172
    label "optimizer"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer"
      fontColor "green"
    ]
  ]
  edge [
    id 168
    source 69
    target 172
    label "lr_scheduler"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler"
      fontColor "green"
    ]
  ]
  edge [
    id 43
    source 70
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 44
    source 71
    target 3
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 45
    source 72
    target 108
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 46
    source 73
    target 170
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 47
    source 74
    target 171
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 48
    source 75
    target 172
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 49
    source 76
    target 9
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 50
    source 77
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 51
    source 78
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 52
    source 79
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 53
    source 80
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 54
    source 81
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 55
    source 82
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 56
    source 83
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 169
    source 83
    target 138
    label "shortcut"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "shortcut"
      fontColor "green"
    ]
  ]
  edge [
    id 57
    source 84
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 170
    source 84
    target 138
    label "shortcut"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "shortcut"
      fontColor "green"
    ]
  ]
  edge [
    id 171
    source 85
    target 55
    label "data"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "data"
      fontColor "green"
    ]
  ]
  edge [
    id 58
    source 86
    target 174
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 59
    source 87
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 60
    source 88
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 61
    source 89
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 62
    source 90
    target 172
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 63
    source 91
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 172
    source 93
    target 14
    label "fc"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "fc"
      fontColor "green"
    ]
  ]
  edge [
    id 173
    source 93
    target 48
    label "classifier"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "classifier"
      fontColor "green"
    ]
  ]
  edge [
    id 174
    source 93
    target 94
    label "orig"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "orig"
      fontColor "green"
    ]
  ]
  edge [
    id 175
    source 93
    target 133
    label "fc"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "fc"
      fontColor "green"
    ]
  ]
  edge [
    id 176
    source 93
    target 154
    label "fc1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "fc1"
      fontColor "green"
    ]
  ]
  edge [
    id 177
    source 93
    target 154
    label "fc2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "fc2"
      fontColor "green"
    ]
  ]
  edge [
    id 178
    source 93
    target 154
    label "fc3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "fc3"
      fontColor "green"
    ]
  ]
  edge [
    id 64
    source 94
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 179
    source 96
    target 59
    label "_logger"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_logger"
      fontColor "green"
    ]
  ]
  edge [
    id 65
    source 97
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 66
    source 98
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 67
    source 99
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 68
    source 100
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 69
    source 101
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 70
    source 102
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 180
    source 103
    target 96
    label "manager"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "manager"
      fontColor "green"
    ]
  ]
  edge [
    id 181
    source 103
    target 96
    label "manager"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "manager"
      fontColor "green"
    ]
  ]
  edge [
    id 71
    source 104
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 72
    source 105
    target 183
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 182
    source 106
    target 154
    label "pool"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "pool"
      fontColor "green"
    ]
  ]
  edge [
    id 183
    source 107
    target 7
    label "attack_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "attack_f"
      fontColor "green"
    ]
  ]
  edge [
    id 184
    source 107
    target 60
    label "model"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "model"
      fontColor "green"
    ]
  ]
  edge [
    id 185
    source 107
    target 60
    label "loss"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "loss"
      fontColor "green"
    ]
  ]
  edge [
    id 186
    source 107
    target 60
    label "eval_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "eval_step"
      fontColor "green"
    ]
  ]
  edge [
    id 187
    source 107
    target 170
    label "weight_decay"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "weight_decay"
      fontColor "green"
    ]
  ]
  edge [
    id 188
    source 107
    target 170
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 189
    source 107
    target 170
    label "epoch_count"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "epoch_count"
      fontColor "green"
    ]
  ]
  edge [
    id 190
    source 107
    target 170
    label "train_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "train_step"
      fontColor "green"
    ]
  ]
  edge [
    id 73
    source 108
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 191
    source 110
    target 66
    label "block_start_columns"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "block_start_columns"
      fontColor "green"
    ]
  ]
  edge [
    id 192
    source 110
    target 66
    label "block_ends"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "block_ends"
      fontColor "green"
    ]
  ]
  edge [
    id 193
    source 110
    target 89
    label "upsample_blends"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "upsample_blends"
      fontColor "green"
    ]
  ]
  edge [
    id 194
    source 110
    target 153
    label "blocks"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "blocks"
      fontColor "green"
    ]
  ]
  edge [
    id 195
    source 111
    target 21
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 196
    source 111
    target 43
    label "info"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "info"
      fontColor "green"
    ]
  ]
  edge [
    id 197
    source 111
    target 53
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 198
    source 111
    target 100
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 199
    source 111
    target 109
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 200
    source 112
    target 21
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 201
    source 112
    target 43
    label "info"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "info"
      fontColor "green"
    ]
  ]
  edge [
    id 202
    source 112
    target 53
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 203
    source 112
    target 100
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 204
    source 112
    target 109
    label "args"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "args"
      fontColor "green"
    ]
  ]
  edge [
    id 74
    source 113
    target 86
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 205
    source 115
    target 110
    label "_modules"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_modules"
      fontColor "green"
    ]
  ]
  edge [
    id 206
    source 115
    target 115
    label "_metadata"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_metadata"
      fontColor "green"
    ]
  ]
  edge [
    id 75
    source 116
    target 9
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 76
    source 117
    target 86
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 207
    source 118
    target 101
    label "block_starts"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "block_starts"
      fontColor "green"
    ]
  ]
  edge [
    id 208
    source 119
    target 93
    label "weight"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "weight"
      fontColor "green"
    ]
  ]
  edge [
    id 209
    source 119
    target 93
    label "bias"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "bias"
      fontColor "green"
    ]
  ]
  edge [
    id 210
    source 121
    target 28
    label "dir_path"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "dir_path"
      fontColor "green"
    ]
  ]
  edge [
    id 211
    source 121
    target 31
    label "_images_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_images_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 212
    source 121
    target 31
    label "_labels_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_labels_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 213
    source 121
    target 44
    label "datasets_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "datasets_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 214
    source 121
    target 79
    label "cache_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "cache_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 215
    source 121
    target 80
    label "_images_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_images_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 216
    source 121
    target 80
    label "_labels_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_labels_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 217
    source 121
    target 81
    label "_data_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_data_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 218
    source 121
    target 178
    label "_images_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_images_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 219
    source 121
    target 178
    label "_labels_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_labels_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 220
    source 121
    target 182
    label "_images_dir"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_images_dir"
      fontColor "green"
    ]
  ]
  edge [
    id 77
    source 122
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 78
    source 123
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 79
    source 125
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 80
    source 126
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 221
    source 127
    target 14
    label "relu"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "relu"
      fontColor "green"
    ]
  ]
  edge [
    id 222
    source 127
    target 15
    label "relu"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "relu"
      fontColor "green"
    ]
  ]
  edge [
    id 223
    source 127
    target 19
    label "relu"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "relu"
      fontColor "green"
    ]
  ]
  edge [
    id 224
    source 127
    target 166
    label "act_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "act_f"
      fontColor "green"
    ]
  ]
  edge [
    id 225
    source 127
    target 186
    label "relu"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "relu"
      fontColor "green"
    ]
  ]
  edge [
    id 81
    source 129
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 82
    source 130
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 83
    source 131
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 84
    source 132
    target 34
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 85
    source 134
    target 132
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 86
    source 135
    target 172
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 87
    source 136
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 88
    source 137
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 89
    source 138
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 226
    source 139
    target 91
    label "upsample"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "upsample"
      fontColor "green"
    ]
  ]
  edge [
    id 90
    source 139
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 227
    source 140
    target 14
    label "root"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "root"
      fontColor "green"
    ]
  ]
  edge [
    id 91
    source 140
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 228
    source 141
    target 59
    label "_logger"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_logger"
      fontColor "green"
    ]
  ]
  edge [
    id 92
    source 141
    target 96
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 93
    source 143
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 94
    source 144
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 95
    source 148
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 96
    source 149
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 229
    source 150
    target 52
    label "spp"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "spp"
      fontColor "green"
    ]
  ]
  edge [
    id 230
    source 150
    target 138
    label "shortcut"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "shortcut"
      fontColor "green"
    ]
  ]
  edge [
    id 231
    source 151
    target 14
    label "layer1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer1"
      fontColor "green"
    ]
  ]
  edge [
    id 232
    source 151
    target 14
    label "layer2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer2"
      fontColor "green"
    ]
  ]
  edge [
    id 233
    source 151
    target 14
    label "layer3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer3"
      fontColor "green"
    ]
  ]
  edge [
    id 234
    source 151
    target 14
    label "layer4"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer4"
      fontColor "green"
    ]
  ]
  edge [
    id 235
    source 151
    target 48
    label "features"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "features"
      fontColor "green"
    ]
  ]
  edge [
    id 236
    source 151
    target 48
    label "features"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "features"
      fontColor "green"
    ]
  ]
  edge [
    id 237
    source 151
    target 133
    label "layer1"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer1"
      fontColor "green"
    ]
  ]
  edge [
    id 238
    source 151
    target 133
    label "layer2"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer2"
      fontColor "green"
    ]
  ]
  edge [
    id 239
    source 151
    target 133
    label "layer3"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer3"
      fontColor "green"
    ]
  ]
  edge [
    id 240
    source 151
    target 133
    label "layer4"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "layer4"
      fontColor "green"
    ]
  ]
  edge [
    id 97
    source 152
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 98
    source 153
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 99
    source 154
    target 108
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 100
    source 155
    target 172
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 241
    source 156
    target 59
    label "state"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "state"
      fontColor "green"
    ]
  ]
  edge [
    id 101
    source 158
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 102
    source 159
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 103
    source 160
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 242
    source 161
    target 66
    label "sum"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "sum"
      fontColor "green"
    ]
  ]
  edge [
    id 243
    source 161
    target 101
    label "sum"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "sum"
      fontColor "green"
    ]
  ]
  edge [
    id 104
    source 161
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 105
    source 162
    target 171
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 244
    source 163
    target 92
    label "initialized"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "initialized"
      fontColor "green"
    ]
  ]
  edge [
    id 245
    source 164
    target 92
    label "initialized"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "initialized"
      fontColor "green"
    ]
  ]
  edge [
    id 246
    source 165
    target 92
    label "initialized"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "initialized"
      fontColor "green"
    ]
  ]
  edge [
    id 106
    source 165
    target 164
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 107
    source 166
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 108
    source 167
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 109
    source 168
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 110
    source 169
    target 86
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 111
    source 170
    target 60
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 112
    source 171
    target 61
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 113
    source 172
    target 62
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 114
    source 176
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 115
    source 177
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 116
    source 178
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 117
    source 179
    target 135
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 118
    source 180
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 119
    source 181
    target 132
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 120
    source 182
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 121
    source 183
    target 109
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 122
    source 184
    target 150
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 123
    source 185
    target 43
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 124
    source 189
    target 158
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 247
    source 190
    target 69
    label "_field_type"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_field_type"
      fontColor "green"
    ]
  ]
  edge [
    id 248
    source 190
    target 69
    label "_field_type"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_field_type"
      fontColor "green"
    ]
  ]
  edge [
    id 249
    source 190
    target 69
    label "_field_type"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_field_type"
      fontColor "green"
    ]
  ]
  edge [
    id 250
    source 192
    target 121
    label "_accessor"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_accessor"
      fontColor "green"
    ]
  ]
  edge [
    id 125
    source 194
    target 151
    graphics [
      targetArrow "empty"
      sourceArrow "none"
    ]
  ]
  edge [
    id 251
    source 195
    target 62
    label "model"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "model"
      fontColor "green"
    ]
  ]
  edge [
    id 252
    source 195
    target 62
    label "loss_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "loss_f"
      fontColor "green"
    ]
  ]
  edge [
    id 253
    source 195
    target 172
    label "weight_decay"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "weight_decay"
      fontColor "green"
    ]
  ]
  edge [
    id 254
    source 195
    target 172
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 255
    source 195
    target 172
    label "epoch_count"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "epoch_count"
      fontColor "green"
    ]
  ]
  edge [
    id 256
    source 196
    target 59
    label "_event_handlers"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_event_handlers"
      fontColor "green"
    ]
  ]
  edge [
    id 257
    source 197
    target 201
    label "dtype"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "dtype"
      fontColor "green"
    ]
  ]
  edge [
    id 258
    source 197
    target 201
    label "dtype"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "dtype"
      fontColor "green"
    ]
  ]
  edge [
    id 259
    source 198
    target 50
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 260
    source 198
    target 50
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 261
    source 198
    target 60
    label "prepare_batch"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "prepare_batch"
      fontColor "green"
    ]
  ]
  edge [
    id 262
    source 198
    target 60
    label "eval_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "eval_step"
      fontColor "green"
    ]
  ]
  edge [
    id 263
    source 198
    target 60
    label "get_outputs"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "get_outputs"
      fontColor "green"
    ]
  ]
  edge [
    id 264
    source 198
    target 60
    label "data_loader_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "data_loader_f"
      fontColor "green"
    ]
  ]
  edge [
    id 265
    source 198
    target 62
    label "prepare_batch"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "prepare_batch"
      fontColor "green"
    ]
  ]
  edge [
    id 266
    source 198
    target 62
    label "data_loader_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "data_loader_f"
      fontColor "green"
    ]
  ]
  edge [
    id 267
    source 198
    target 90
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 268
    source 198
    target 90
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 269
    source 198
    target 135
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 270
    source 198
    target 135
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 271
    source 198
    target 155
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 272
    source 198
    target 170
    label "train_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "train_step"
      fontColor "green"
    ]
  ]
  edge [
    id 273
    source 198
    target 170
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 274
    source 198
    target 172
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 275
    source 199
    target 50
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 276
    source 199
    target 50
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 277
    source 199
    target 60
    label "prepare_batch"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "prepare_batch"
      fontColor "green"
    ]
  ]
  edge [
    id 278
    source 199
    target 60
    label "eval_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "eval_step"
      fontColor "green"
    ]
  ]
  edge [
    id 279
    source 199
    target 60
    label "get_outputs"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "get_outputs"
      fontColor "green"
    ]
  ]
  edge [
    id 280
    source 199
    target 60
    label "data_loader_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "data_loader_f"
      fontColor "green"
    ]
  ]
  edge [
    id 281
    source 199
    target 62
    label "prepare_batch"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "prepare_batch"
      fontColor "green"
    ]
  ]
  edge [
    id 282
    source 199
    target 62
    label "data_loader_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "data_loader_f"
      fontColor "green"
    ]
  ]
  edge [
    id 283
    source 199
    target 90
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 284
    source 199
    target 90
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 285
    source 199
    target 135
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 286
    source 199
    target 135
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 287
    source 199
    target 155
    label "optimizer_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "optimizer_f"
      fontColor "green"
    ]
  ]
  edge [
    id 288
    source 199
    target 170
    label "train_step"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "train_step"
      fontColor "green"
    ]
  ]
  edge [
    id 289
    source 199
    target 170
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 290
    source 199
    target 172
    label "lr_scheduler_f"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "lr_scheduler_f"
      fontColor "green"
    ]
  ]
  edge [
    id 291
    source 200
    target 42
    label "__init__"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "__init__"
      fontColor "green"
    ]
  ]
  edge [
    id 292
    source 200
    target 47
    label "__init__"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "__init__"
      fontColor "green"
    ]
  ]
  edge [
    id 293
    source 200
    target 132
    label "__init__"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "__init__"
      fontColor "green"
    ]
  ]
  edge [
    id 294
    source 200
    target 134
    label "__init__"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "__init__"
      fontColor "green"
    ]
  ]
  edge [
    id 295
    source 200
    target 181
    label "__init__"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "__init__"
      fontColor "green"
    ]
  ]
  edge [
    id 296
    source 201
    target 30
    label "y"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "y"
      fontColor "green"
    ]
  ]
  edge [
    id 297
    source 201
    target 30
    label "y"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "y"
      fontColor "green"
    ]
  ]
  edge [
    id 298
    source 201
    target 144
    label "_indices"
    graphics [
      targetArrow "diamond"
      sourceArrow "none"
      style "line"
    ]
    LabelGraphics [
      text "_indices"
      fontColor "green"
    ]
  ]
]
