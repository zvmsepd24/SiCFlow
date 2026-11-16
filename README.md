# Convergence of Flow Models Trained on a SiC/SiC Fiber Composite

Code for the paper *Convergence of Flow Models Trained on a SiC/SiC
Fiber Composite*. The section comes from the S200 SiC/SiC serial-section data set (Materials Data Facility,
doi:10.18126/M2135G; Sherman et al., Acta Materialia 181, 447-459, 2019).

Flow-matching models are trained on one polished section of a SiC/SiC composite. Paired models are trained
on the two colours of a checkerboard of tiles, which gives two statistically equivalent training sets, at
training areas from 0.02 to 3.2 mm² per set. Ensembles of generated windows are compared with held-out
windows (fidelity) and with each other (convergence) through descriptors of the segmented windows
(two-point function, chord-length distributions, pore fraction, fiber diameter) and the Fréchet Inception
distance, and the Levina–Bickel intrinsic dimension of clouds of windows and descriptors is estimated.
Models trained on the whole training band, including a two-scale model, are then compared on fields much
larger than their training crops.

## Layout

- `genmicro/`: package code
  - `sic.py`: loading the section and cropping
  - `checkerboard.py`: the checkerboard split, nested training areas and similarity to training tiles
  - `net.py`: the fully convolutional U-Net
  - `train.py`: flow-matching training (Muon and AdamW) and the Heun sampler
  - `twoscale.py`: layout fields (the state layout of local fiber and pore fractions, and the earlier
    gray-level layout), the conditioned fiber model, noise on the layout (conditioning augmentation), and
    two-scale sampling
  - `phases.py`: segmentation into fibers, matrix and pores
  - `descriptors.py`: two-point functions and chord lengths
  - `fibers.py`: fiber diameters and Delaunay statistics
  - `scores.py`: Inception and DINOv2 features, Fréchet distance, kernel inception distance, loading models
  - `lid.py`: the Levina–Bickel estimator, and a flow-based local intrinsic dimension estimate that is not used in the paper
- `scripts/`: experiments, figures and tables, listed below
- `results/sic/`: written by the scripts: numerical results (JSON), sample arrays (`*.npz`) and
  checkpoints (`*.pt`). The figure scripts read them.
- `data/sic/`: the section data (uploaded separately from the code):
  - `section_plane.npy`: the training section at 1.04 µm per pixel (8-bit gray levels, 2376 x 3597
    pixels), written by `scripts/prepare_sic.py` from the stitched image of the S200 data set. It is
    section 10 of the series (`Sample1_10`). Every script reads it.
  - `sections/section99.npy` and `sections/section99_mask.npy`: section 99 of the series, 89 sections
    further into the specimen, stitched from its 6 x 6 tiles (`scripts/stitch_section.py`), registered to
    the training section and cut in its frame, with the mask of the pixels its montage covers
    (`scripts/register_sections.py`); `section10.npy` and its mask: the training section stitched again
    the same way, a check of the stitching. Only the scripts of step 8 read them.
- `figures/`, `tables/`: written by the figure and table scripts

## Reproducing the results

Python 3.14 and a CUDA GPU; versions in `requirements.txt`. Two trainings can share a 12 GB GPU. Training
is not bitwise reproducible (compiled kernels and nondeterministic GPU reductions), so retrained models
differ in detail from those in the paper.

1. **Prepare the section** (provided separately; to rebuild `data/sic/section_plane.npy` from the stitched image):

   ```bash
   python scripts/prepare_sic.py PATH_TO_STITCHED_TIFF
   ```

2. **Checkerboard models and their evaluation**, for replicates 0, 1 and 2:

   ```bash
   python scripts/train_checkerboard.py 0
   python scripts/evaluate_checkerboard.py 0
   ```

3. **Models of the whole training band.**

   ```bash
   python scripts/train_single_scale.py 64      # and 128, 256
   python scripts/train_two_scale.py 1          # seeds 1, 2, 3; layout and fiber model
   python scripts/evaluate_band_models.py       # all models, or a list of names
   python scripts/evaluate_layout_sampler.py    # fiber model on real and generated layouts (Table S5)
   ```

4. **Figures and tables.** Each script writes to `figures/` or `tables/`.

   The paper has five main figures; the scripts of steps 5 and 6 must have run for Figs. 3–5.

   | Script | Output |
   |---|---|
   | `fig_setup.py` | Fig. 1: the section, the checkerboard, the two scales |
   | `fig_generalization.py` | Fig. 2: copying, same-noise agreement and intrinsic dimension against training area |
   | `fig_model_gap.py` | Fig. 3: convergence of descriptors; more data, sampler, network and training |
   | `fig_tow_structure.py` | Fig. 4: tow bands, copying at the tow scale, copying against the number of crops |
   | `fig_tow_window.py` | Fig. 5: the paired test at the tow scale; the two-scale model |
   | `fig_memorization.py` | supplement: copying at all five areas |
   | `fig_convergence.py` | supplement: all descriptors; intrinsic dimension of pixel and chord clouds |
   | `fig_dimension_crops.py` | supplement: dimension of real windows against the number of crops, both scales |
   | `fig_examples.py` | fields of the whole-band models (earlier figure; replaced by `fig_tow_fields.py`) |
   | `fig_structure.py` | two-point and chord statistics along and across the bands (supplement) |
   | `fig_statistics.py` | FID, fiber and pore statistics of the whole-band models (supplement) |
   | `supplement_tables.py` | Tables S1–S6 |

Every model is sampled with one sampler, 32 Heun steps uniform in t (`STEPS`, `RHO` in `genmicro/train.py`).

5. **Checks of the 64-pixel model and same-noise agreement** (the 64-pixel model is the model of the whole
   training band, all of the training data; run GPU jobs one at a time on a 12 GB card).

   ```bash
   python scripts/train_single_scale.py 64 --seed 1         # second training seed
   python scripts/train_single_scale.py 64 --wide           # 1.5 times the channels
   python scripts/train_single_scale.py 64 --steps 12000    # 1.5 times the training steps
   python scripts/train_checkerboard.py 0 --scratch         # paired models of 3.2 mm^2 trained from scratch, 8000 steps
   python scripts/evaluate_windows.py single64 single64@64r1 A4 B4   # 64 px windows; NAME@STEPS[rRHO][nN] sets the sampler
   python scripts/disagreement.py                           # same-noise agreement per sample, paired models
   python scripts/disagreement.py single64_seed1,single64   # the 64-pixel model, two seeds
   ```

   | Script | Output |
   |---|---|
   | `fig_disagreement.py` | where the paired models disagree from the same noise |
   | `fig_data_model.py` | more training data against changes of the model |
   | `table_model_checks.py` | table of the checks of the 64-pixel model |


6. **The tow scale** (fiber-fraction layout; results in `results/sic/towscale/`). The two-scale model of the paper
   is the layout model of the whole band stopped after 1000 steps (`layouts/steps1000_s{1,2,3}`) and the fiber model
   trained with noise on the layout and stopped after 2000 steps (`fiber_stateaug2k_s{1,2,3}`); in the scripts it is
   `twoscale_early_2k_s{1,2,3}` (`TWO_SCALE` in `scripts/figstyle.py`). The same fiber model after 1000, 4000, 6000,
   8000 and 16000 steps (`stateaug1k`, `stateaugshort`, `stateaug6k`, `stateaug`, `stateauglong`; `twoscale_earlyshort_s*`,
   `twoscale_early_{1k,6k,long}_s1`, `twoscale_early_s*`) is the training-length check of the supplement.

   ```bash
   python scripts/train_two_scale.py 1 stateaug2k           # seeds 1, 2, 3: fiber model, 2000 steps
   python scripts/train_layout.py steps1000_s1 --steps 1000 # seeds 1, 2, 3: the layout model of the two-scale model
   python scripts/evaluate_band_models.py twoscale_early_2k_s1 twoscale_early_2k_s2 twoscale_early_2k_s3
   python scripts/train_two_scale.py 1 stateaugshort        # check: 4000 steps (seeds 1-3)
   python scripts/train_two_scale.py 1 stateaug             # check: fiber model after 8000 steps (and a 6000-step layout model)
   python scripts/train_two_scale.py 1 stateauglong         # check: fiber model after 16000 steps
   python scripts/train_two_scale.py 1 stateaug1k           # check: 1000 steps (also stateaug6k), seed 1;
                                                            # evaluate as twoscale_early_{1k,6k}_s1; pairs: evaluate_fiber_pairs.py stateaug2k
   python scripts/train_two_scale.py 1 state                # check: fiber model trained on exact layouts
   python scripts/train_single_scale.py 512                 # 512 px crops, deeper network
   python scripts/train_layout.py A_s1 --cols 0:1320        # layout models: halves (A, B), a quarter, crop sizes 16-64
   python scripts/train_layout.py A500_s1 --cols 0:1320 --steps 500   # the paired test against training steps
   python scripts/train_two_scale.py 1 stateblur            # check: noise blurred beyond the fiber scale
   python scripts/evaluate_band_models.py twoscale_mix_l5   # check: velocity mixed with the 128 px model (l3, l5, l7)
   python scripts/train_two_scale.py 1 stateaugshort --cols 0:1320 --tag A   # fiber model on one half (paired test;
                                                            # also B on 1320:2640, seed 2 on A, and stateaug, stateauglong)
   python scripts/evaluate_tow_scale.py single64 twoscale_early_2k_s1 layouts/A_s1 layouts/B_s1   # tow statistics, copying
   python scripts/evaluate_tow_dimension.py                 # intrinsic dimension of 0.53 mm layout windows (Fig. 5e)
   python scripts/evaluate_fiber_given_layout.py fiber_state_s1 fiber_stateaug2k_s1@0.5          # fiber models on real layouts
   python scripts/evaluate_fiber_pairs.py stateaugshort     # fiber models of the two halves from the same noise
                                                            # (no argument: stateaug, 8000 steps; also stateauglong)
   python scripts/key_numbers_tow.py                        # numbers quoted in the text
   ```

   | Script | Output |
   |---|---|
   | `fig_tow_fields.py` | supplement: tow statistics of every model, along and across |
   | `fig_tow_copying.py` | supplement: copying at the tow scale per model (also `scaling`, used by Fig. 4) |
   | `fig_tow_paired.py` | earlier figure of the paired test at the tow scale (now part of Fig. 5) |
   | `fig_fiber_given_layout.py` | supplement: the fiber model given a layout: copying without noise on the layout; fidelity against training steps |


7. **Checks of the test across models** (results in `results/sic/families/` and `results/sic/convergence/`).
   Other families of generative models on the same checkerboard, with the curriculum of the paired models: denoising
   diffusion (DDPM with the U-Net of the flow models, sampled with deterministic DDIM), a variational autoencoder
   (sigma-VAE) and a normalizing flow (Glow); `genmicro/families.py`. Each family trains models A, B and A2 (set A,
   another seed). The flow models are continuous normalizing flows, so their likelihood of training and held-out
   windows can be computed as for Glow. Paired flow models of half and 1.5 times the width test whether the area at
   which copying stops depends on the size of the network.

   ```bash
   python scripts/train_family.py vae 0          # also ddpm, glow; replicate 0 (vae also 1, 2; ddpm also 1)
   python scripts/evaluate_family.py vae 0       # copying, convergence, fidelity, agreement; Glow: bits per dimension
   python scripts/evaluate_family.py glow 0 --temperature 0.7   # Glow sampled at a lower temperature
   python scripts/evaluate_likelihood.py 0 1 2 single64   # bits per dimension of the flow models, training vs held-out
   python scripts/train_checkerboard.py 0 --widths 32,64,128,128 --tag w05     # and 96,192,384,384 --tag w15
   python scripts/evaluate_checkerboard.py 0 --tag w05                         # -> convergence/width_w05_rep0.json
   ```

   | Script | Output |
   |---|---|
   | `fig_families.py` | supplement: the paired test for other model families, and the likelihood gap |
   | `fig_width.py` | supplement: the paired test for half and 1.5 times the network width |
   | `integral_range.py` | supplement: integral range of the fiber and pore fractions of the section (representative area) |

8. **Section 99 of the specimen** (results in `results/sic/sections/`). The S200 series holds further sections of the
   same specimen. Section 99, 89 sections further in, has other fibers but the same plies; its left third is out of
   focus, so only its right half is used. It is used only as a reference, never for training: scored against the
   held-out band like a model (a yardstick for the scores of the models), and as an independent test set for the
   paired models.

   ```bash
   python scripts/stitch_section.py TILE_DIR data/sic/stitched/slice99.png         # section 99 (tiles 00.png-35.png)
   python scripts/stitch_section.py TILE_DIR data/sic/stitched/slice10.png --pattern "Sample1_10_p{i:02d}.tif"
                                                  # section 10 stitched again: a check of the stitching
   python scripts/register_sections.py 10 99      # -> data/sic/sections/section{NN}.npy
   python scripts/compare_sections.py 10 99       # fibers and tows of section 99 against the training section
   python scripts/evaluate_sections.py            # section 99 scored against the held-out band
   python scripts/evaluate_against_sections.py    # real sets and band models against the held-out band and section 99
   python scripts/evaluate_independent.py --real  # distances between the real sets
   python scripts/evaluate_independent.py 0 1 2   # paired models against section 99
   ```

   | Script | Output |
   |---|---|
   | `fig_independent.py` | supplement: section 99 as yardstick and as independent test set |
