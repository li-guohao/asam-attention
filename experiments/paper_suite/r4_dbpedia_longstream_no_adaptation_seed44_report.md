# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Output classes: `14`
- Avg accuracy: `0.1036`
- Avg forgetting: `0.3750`
- Backward transfer: `-0.2417`

## Artifacts

- Plot image: `r4_dbpedia_longstream_no_adaptation_seed44_plots.png`
- Raw JSON: `r4_dbpedia_longstream_no_adaptation_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.8404`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.375, 0.4166666666666667, 0.4, 0.37, 0.375]`
- Stage transport gap trace: `[0.04704268276691437, 0.0462835393846035, 0.04566429369151592, 0.04558341205120087, 0.045376600697636604, 0.041350703686475754, 0.043878912925720215]`
- Stage transport loss trace: `[0.17218391249577206, 0.09351337353388468, 0.07651349430282911, 0.07405583560466766, 0.0580661877989769, 0.07834436744451523, 0.08454501784096162]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.01647813990712166, 0.21394655108451843, 0.2040623500943184, 0.20713059604167938, 0.20054014772176743, 0.20163267850875854, 0.19639339298009872]`
- Stage Birkhoff applied-offdiag trace: `[0.00032956279814243315, 0.004278931021690368, 0.0040812470018863675, 0.004142611920833587, 0.004010802954435348, 0.004032653570175171, 0.0039278678596019745]`
- Stage Birkhoff gap-delta trace: `[-1.7309561371803284e-05, -0.000321732833981514, -0.00029287301003932953, -0.00030321069061756134, -0.00031653791666030884, -0.00027049146592617035, -0.0003003198653459549]`
- Stage Birkhoff row-error trace: `[6.985664367675781e-05, 1.1920928955078125e-07, 1.7881393432617188e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.053996566062172256, 0.7851994434992472, 0.6566091020901997, 0.578823717435201, 0.48948232730229696, 0.6448233922322592, 0.5992826223373413]`
- Forgetting vs routing stability correlation: `0.34889164692575525`
- Forgetting vs transport gap correlation: `-0.5489289124392932`
- Forgetting vs transport loss correlation: `-0.7717326712586464`
- Forgetting vs mean abs excess correlation: `-0.548929443964487`
- Forgetting vs merge-count correlation: `0.6672288922736431`