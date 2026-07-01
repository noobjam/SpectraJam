# Research record

Generated 2026-07-01 from primary papers, official repositories, official data
documentation, and the local TesseraCrop audit. Confidence is high for TESSERA
v1.0 and ingestion contracts, medium for undocumented v1.1 training behavior,
and moderate for the final regional sample count.

## Core TESSERA evidence

- [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_TESSERA_Temporal_Embeddings_of_Surface_Spectra_for_Earth_Representation_and_CVPR_2026_paper.html)
- [CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Feng_TESSERA_Temporal_Embeddings_CVPR_2026_supplemental.pdf)
- [Official repository](https://github.com/ucam-eo/tessera/tree/d06ee44a053246db3e73f104403f6eaf642e1abf)
- [Earlier sparse temporal Barlow Twins work](https://doi.org/10.1109/JSTARS.2024.3426044)

TESSERA v1.0 trained on about 800 million annual d-pixels from 3,012 MGRS tiles
over 2017–2024. It thinned pixels by 20 in each spatial axis, sampled two
independent 40-observation temporal views, used Barlow Twins plus mixup, and
globally shuffled samples. Removing global shuffling or mixup reduced Austrian
crop F1 by 9.2 and 11.1 points respectively. Its paper reports negligible gain
from region-specific retraining, which makes a frozen baseline mandatory.

The current v1.1 release differs from the paper: width 768, 192-D reducer with
the first 128 dimensions stored, all-observation bucketed inference, separate
ascending/descending S1 normalization, and distinct MPC/AWS checkpoints. Its
full executable pretraining recipe is not published, so paper-equivalence must
not be claimed for v1.1 regional SSL.

The MPC encoder artifact is pinned to Hugging Face repository revision
`e037fc62cd196f9e05dde4c4104e1383541b41c5`, 230,891,229 bytes, with observed
SHA-256 `5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3`.
Upstream does not publish a checksum, so this is recorded as an independently
observed release hash rather than an upstream checksum. The downloader refuses
to accept an unnoticed artifact replacement.

## Regional and parameter-efficient adaptation

- [GDA: Parameter Efficient Self-Supervised Geospatial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/papers/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.pdf)
- [ExPLoRA](https://arxiv.org/html/2406.10973)
- [IBM PEFT-GeoFM experiments](https://github.com/IBM/peft-geofm)
- [Official Microsoft LoRA merged-QKV guidance](https://github.com/microsoft/LoRA#additional-notes)
- [Continual geospatial pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.pdf)
- [TerraFM sampling design](https://arxiv.org/html/2506.06281)
- [Presto structured temporal/channel masking](https://arxiv.org/html/2304.14065)
- [DINO-MM modality dropout](https://arxiv.org/abs/2204.05381)

These works support unlabeled continual pretraining with small adapters,
regional SSL in addition to teacher matching, packed Q/V LoRA, structured
temporal masking, modality dropout, and land-cover/climate-aware sampling. They
do not establish that LoRA improves TESSERA specifically; that remains the
experiment.

## Arbitrary-window and early-decision evidence

- [ELECTS: End-to-End Learned Early Classification for In-Season Crop Type Mapping](https://arxiv.org/abs/1901.10681)
- [Early Classification for Agricultural Monitoring from Satellite Time Series](https://arxiv.org/abs/1908.10283)
- [TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/download/20881/20640)
- [Seasonal Contrast](https://openaccess.thecvf.com/content/ICCV2021/papers/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.pdf)
- [AnySat](https://openaccess.thecvf.com/content/CVPR2025/papers/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.pdf)
- [data2vec](https://arxiv.org/abs/2202.03555)
- [Retentive Network](https://arxiv.org/abs/2307.08621)
- [Transformer-XL](https://aclanthology.org/P19-1285/)
- [EarthPT](https://arxiv.org/abs/2309.07207)

Early crop-classification work supports making predictions from prefixes using
only observations available at the cutoff. It does not establish that 7-day or
14-day foundation embeddings will be accurate. The strongest TESSERA
short-window result is seasonal-scale (roughly nine months), while performance
drops sharply at low valid-observation counts. Seven and fourteen days are
therefore required evaluation anchors, not claims inherited from TESSERA.

TS2Vec cautions against treating every cropped subseries as semantically
identical and aligns overlapping temporal contexts more carefully. Seasonal
Contrast likewise separates seasonally stable and seasonally varying signals.
These results support same-window SSL and preserving dynamic phenology rather
than forcing the exported short-window embedding to equal one annual vector.
AnySat, data2vec, and Presto support masked/privileged-view objectives, but
those are denoising or imputation pretexts; a full-context target must not be
described as a purely observed-state target.

The initial SpectraJam implementation consequently uses a frozen TESSERA target
from the same exact window and scales its influence by duration and actual
observation evidence. A later ablation may add an annual-semantic prediction
head used only during training. If added, it must be named as forecasting or
privileged-context distillation and must not directly collapse all nested
windows to one annual representation.

RetNet and Transformer-XL demonstrate that efficient recurrent inference
requires an architecture designed for state reuse. TESSERA's bidirectional
attention and GRU pooling do not have an exact add/subtract update for a strict
rolling window. EarthPT is a useful EO precedent: despite causal modeling, its
reference path crops and recomputes the active block. SpectraJam therefore
defines incremental v1 as input-aware caching plus exact active-window replay;
a causal recurrent student is a separate future model with a mandatory parity
evaluation.

## Data and boundary sources

- [ESA global Sentinel-2 L2A start](https://sentinels.copernicus.eu/-/copernicus-sentinel-2-level-2a-production-goes-global)
- [Microsoft Planetary Computer STAC](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)
- [Copernicus Data Space STAC](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)
- [OGC STAC Item Search](https://docs.ogc.org/cs/25-005/25-005.html)
- [ESA WorldCover 2021](https://worldcover2021.esa.int/)
- [RESOLVE Ecoregions 2017](https://developers.google.com/earth-engine/datasets/catalog/RESOLVE_ECOREGIONS_2017)
- [Copernicus DEM](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html)
- [WorldClim 2](https://www.fao.org/land-water/land/land-governance/land-resources-planning-toolbox/category/details/en/c/1043064/)
- [World Bank Official Boundaries v2](https://datacatalog.worldbank.org/infrastructure-data/search/dataset/0038272/world-bank-official-boundaries)
- [UN SALB boundary disclaimer](https://salb.un.org/en)

## Local audit result

TesseraCrop cannot be treated as the canonical base. It uses the conventional
S2 order rather than the official checkpoint order, non-official normalization
statistics, an early 8-layer/8-head topology, and permissive checkpoint loading.
SpectraJam therefore starts from a clean repository with explicit compatibility
contracts and an upstream parity receipt.
