# QRS detection using K-Nearest Neighbor algorithm (KNN) and evaluation on standard ECG databases
## Introduction
The function of human body is frequently associated with signals of electrical, chemical, or acoustic origin. Extracting useful information from these biomedical signals has been found very helpful in explaining and identifying various pathological conditions. The most important are the signals which are originated from the heart's electrical activity. This electrical activity of the human heart, though it is quite low in amplitude (about 1 mV) can be detected on the body surface and recorded as an electrocardiogram (ECG) signal. The ECG arise because active tissues within the heart generate electrical currents, which flow most intensively within the heart muscle itself, and with lesser intensity throughout the body. The flow of current creates voltages between the sites on the body surface where the electrodes are placed. The normal ECG signal consists of P, QRS and T waves. The QRS interval is a measure of the total duration of ventricular tissue depolarization. QRS detection provides the fundamental reference for almost all automated ECG analysis algorithms. Before to perform QRS detection, removal or suppresion of noise is required. The aim of this work is to explore the merits of KNN algorithm as an ECG delineator. The KNN method is an instance based learning method that stores all available data points and classifies new data points based on similarity measure. In KNN, the each training data consists of a set of vectors and every vector has its own positive or negative class label, where K represents the number of neighbors. 

## Dependencies
The modules are implemented for use with Python 3.x and they consist of the following dependencies:
* scipy
* numpy
* matplotlib
```
pip install wfdb
```

## Repository directory structure 
```
├── Main.py
|
├── FeatureExtraction.py
|
├── GridSearch.py
|
├── MultiClassClassification.py
|
└── README.md
```

## Installation and Usage

## MIT-BIH Arrhythmia Database
The MIT-BIH Arrhytmia DB contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects (records 201 and 202 are from the same subject) studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Of these, 23 were chosen at random from a collection of over 4000 Holter tapes, and the other 25 (the "200 series") were selected to include examples of uncommon but clinically important arrhythmias that would not be well represented in a small random sample. Approximately 60% of the subjects were inpatients. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range. The digitization rate was chosen to accommodate the use of simple digital notch filters to remove 60 Hz (mains frequency) interference. Six of the 48 records contain a total of 33 beats that remain unclassified, because the cardiologist-annotators were unable to reach agreement on the beat types. The annotators were instructed to use all evidence available from both signals to identify every detectable QRS complex. The database contains seven episodes of loss of signal or noise so severe in both channels simultaneously that QRS complexes cannot be detected; these episodes are all quite short and have a total duration of 10s. In all of the remaining data, every QRS complex was annotated, about 109.000 in all. 

For further descriptions, please see the references. 

### Annotations 
|    Beat Annotation     |            Meaning             |
| ------------- | -------------------------------|
|N|Normal Beat|
|L|Left bundle branch block beat|
|R|Right bundle branch block beat|
|B|Bundle branch block beat (unspecified)|
|A|Atrial premature beat|
|a|Aberrated atrial premature beat|
|J|Nodal (junctional) premature beat|
|S|Supraventricular premature beat|
|V|Premature ventricular contraction|
|r|R-on-T premature ventricular contraction|
|F|Fusion of ventricular and normal beat|
|e|Atrial escape beat|
|j|Nodal (junctional) escape beat|
|n|Supraventricular escape beat (atrial or nodal)|
|E|Ventricular escape beat|
|/|Paced beat|
|f|Fusion of paced and normal beat|
|Q|Unclassifiable beat|
|?|Beat not classified during learning|

|Non-Beat Annotation | Meaning| 
|--------------------|--------|
|\[|Start of ventricular flutter/fibrillation|
|!|Ventricular flutter wave|
|\]|End of Ventricular flutter/fibrillation|
|x|Non-conducted P-wave (blocked APB)|
|(|Waveform onset|
|)|Wafeform end|
|p|Peak of P-wave|
|t|Peak of T-wave|
|u|Peak of U-wave|
|\`|PQ junction|
|'|J-point|
|^|(Non-captured) pacemaker artifact|
|\|| Isolated QRS-like artifact|
|~|Change in signal quality|
|+|Rhythm change|
|s|ST segment change|
|T|T-wave change|
|\*|Systole|
|D|Diastole|
|=|Measurement annotation|
|"|Comment annotation|
|@|Link to external data|


## The Algorithm 
The algorithm is actually divided in three main phases : Read the data, Extract the features and Classification. 

### Reading Data
### Feature Extraction
#### Two-Class Labels
#### Multi-Class labels
### KNN Classifier
#### Parameters
### Output

## References 
* [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 


