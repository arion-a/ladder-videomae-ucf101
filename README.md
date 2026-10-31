# ladder-videomae-ucf101
Action - perception feature extraction comparison over standard UCF101 result,  between my self created ladder( for feature extraction) and Video MAE V2 model

## Ladder of Invariants vs frozen VideoMAE V2 (UCF101 split 1)

Question: does the ladder's representation beat VideoMAE V2's own frozen features for linear-probe action recognition, under one protocol?
- Encoder: `OpenGVLab/VideoMAEv2-Base` (self-supervised), frozen, fp32.
- Per video: 16 uniformly spaced frames -> one clip -> 8 time vectors x 768 (spatial mean of final-block tokens, then the model's own fc_norm). The model's native pooled vector is saved as a reference row.
- Ladder: `ladder_final.py` (unchanged), lags 1 and 2 in token steps, theta 0.5, fitted on train groups 8-21 only. Validation = train groups 22-25 (held out by group). Test = official `testlist01`.
- Arms: native vector; mean; mean+std; PCA to the ladder's dimension then mean+std; ladder level (chosen on validation). Control: ladder fitted on time-shuffled sequences.
- Decision: paired bootstrap over test videos, 95% interval of the accuracy difference.
Run: open `run_colab.ipynb` in Colab (T4 GPU) and Run all. `extract.py` makes the features, `fit_probe.py` does everything else.
Tested locally only on synthetic features (the code path runs end to end); extraction has not yet been run on real videos.
