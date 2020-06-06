# V1_ECG_Classifier_CNN
Making ECG classfier with CNN and V1 lead ECG graph  
classify ecg into SINUS, RBBB, LBBB, PVC, VT, NSVT
## progresss
- Segmenting ECG time seriese in segments  **DONE**
    - Get R peak using pan-tompkin’s algorithm
    - Centering R peak and segmenting left and right around
- Labeling each segment **DONE**
- Make multilabel classifier model **DONE**
- Training **DONE**
- Nexting to do
- clear up project

## Current Result
I got about 90% train set acc and test set acc  
The Problem is the ambiguous ECG
## Next
to detect ambiguous ECG, i will try Ensemble.  
make each 12-lead-ECG to CNN model and Ensemble.

#### compare docs uploaded