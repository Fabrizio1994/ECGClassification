# QRS detection using K-Nearest Neighbor algorithm (KNN) and evaluation on standard ECG databases
## Introduction
The function of human body is frequently associated with signals of electrical, chemical, or acoustic origin. Extracting useful information from these biomedical signals has been found very helpful in explaining and identifying various pathological conditions. The most important are the signals which are originated from the heart's electrical activity. This electrical activity of the human heart, though it is quite low in amplitude (about 1 mV) can be detected on the body surface and recorded as an electrocardiogram (ECG) signal. The ECG arise because active tissues within the heart generate electrical currents, which flow most intensively within the heart muscle itself, and with lesser intensity throughout the body. The flow of current creates voltages between the sites on the body surface where the electrodes are placed. The normal ECG signal consists of P, QRS and T waves. The QRS interval is a measure of the total duration of ventricular tissue depolarization. QRS detection provides the fundamental reference for almost all automated ECG analysis algorithms. Before to perform QRS detection, removal or suppresion of noise is required. The aim of this work is to explore the merits of KNN algorithm as an ECG delineator. The KNN method is an instance based learning method that stores all available data points and classifies new data points based on similarity measure. In KNN, the each training data consists of a set of vectors and every vector has its own positive or negative class label, where K represents the number of neighbors. 

## Dependencies
The modules are implemented for use with Python 3.x and they consist of the following dependencies:
* scip
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
├── TwoClassClassification.py
|
└── README.md
```

## MIT-BIH Arrhythmia Database
The MIT-BIH Arrhytmia DB contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects (records 201 and 202 are from the same subject) studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Of these, 23 were chosen at random from a collection of over 4000 Holter tapes, and the other 25 (the "200 series") were selected to include examples of uncommon but clinically important arrhythmias that would not be well represented in a small random sample. Approximately 60% of the subjects were inpatients. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range. The digitization rate was chosen to accommodate the use of simple digital notch filters to remove 60 Hz (mains frequency) interference. Six of the 48 records contain a total of 33 beats that remain unclassified, because the cardiologist-annotators were unable to reach agreement on the beat types. The annotators were instructed to use all evidence available from both signals to identify every detectable QRS complex. The database contains seven episodes of loss of signal or noise so severe in both channels simultaneously that QRS complexes cannot be detected; these episodes are all quite short and have a total duration of 10s. In all of the remaining data, every QRS complex was annotated, about 109.000 in all. 

### QRS Region
The QRS complex is the central part of an ECG. It corresponds to the depolarization of the right and left ventricles of the human heart. A Q wave is any downward deflection after the P wave. An R wave follows as an upward deflection, and the S wave is any downward deflection after the R wave. Since each annotation we got from the MIT-BIH Database is not precisely associated to an R-Peak but to a whole QRS complex, we had to consider a window of fixed size (±5, ±10, ±25) with the R-peak as center point, considering as good annotations those ones that will lay inside the window. An example of a PQRST segment can be seen in the picture below. 

![QRS complex](https://preview.ibb.co/htTGsn/PQRST.png)

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
The aim of this work is to provide two efficient solutions for the problems of QRS detection and multiclass classification. 
Every signal is annotated by cardiologists with the locations of the QRS complexes, and labeled with a symbols from the list above.  
In this scope, the QRS detection problem is encountered as a binary classification problem:
each sample point of a given signal is considered as a feature that can be classified either as a QRS complex or not.  
The before mentioned QRS multiclass classification aims to label each sample point with one of the symbols listed in the Annotations section. 

Each signal is considered for 80% of its length as training data and the rest as test data.
The training data of the multiclass problem is composed by the first 20.000 samples of each of the 48 record in the MIT-BIH database.
The algorithm is actually divided in four main phases : Reading data, Signal Processing, Feature Extraction and Classification. 

### Reading Data
Data are available in the PhysioNet website, precisely at the link below:

https://www.physionet.org/physiobank/database/mitdb/  
Dataset is divided in three standard categories: 

* MIT Signal files (.dat) are binary files containing samples of digitized signals. These store the waveforms, but they cannot be interpreted properly without their corresponding header files. These files are in the form: RECORDNAME.dat.
* MIT Header files (.hea) are short text files that describe the contents of associated signal files. These files are in the form: RECORDNAME.hea.
* MIT Annotation files are binary files containing annotations (labels that generally refer to specific samples in associated signal files). Annotation files should be read with their associated header files. If you see files in a directory called RECORDNAME.dat, or RECORDNAME.hea, any other file with the same name but different extension, for example RECORDNAME.atr, is an annotation file for that record.


Raw signals are loaded with rdsamp function from WFDB package:
```
wfdb.rdrecord(path_to_sample, samp_from, samp_to)
```
where path_to_sample is the local path where the records are stored, and samp_from and samp_to define the portion of signal, contained in a range of frequencies, considered for processing.
Each record in the database comprehends two raw signals, coming from the two channels of ECG recording. 
### Signal Processing
Signals are processed by using a band pass filter, in order to reduce the recording noise that would lead to uncorrect classifications.  
The high pass filter chosen is the one suggested in "QRS detection using KNN" paper, defined by the following transfer function:  
![equation](http://latex.codecogs.com/gif.latex?H(z)&space;=&space;\frac{-1&plus;32z^{-16}&plus;z^{-32}}{1&plus;z^{-1}})  
This step is implemented by means of the lfilter function of scipy package:  
```
output_record = signal.lfilter(num_coefficents, den_coefficients, input_record)
```
where num_coefficients and den_coefficients are the lists of exponent values of the transfer function of the numerator and the denominator respectively.  


![Non-Filtered Signal](https://image.ibb.co/gCiVgS/prefiltered100.png)
![Filtered Signal](https://image.ibb.co/f4pQFn/filtered100.png)

### Feature Extraction
The KNN classifier expects as input a feature vector for each sample point.
Values in such vector should describe the signal function trend with the purpose of detecting peaks.  
Therefore this step requires the computation of the gradient vector of the two signals of each record, considering its elements as features. In this way each sample corresponds to a 2D feature vector. 
#### Two-Class Labels
In the binary classification scope, each feature has assigned a value {-1,1} whether the original sample point results annotated as a peak.  
#### Multi-Class labels
Multiclass labels are defined with an integer value in (0,38) corresponding to annotations symbols and -1 for samples that aren't peaks.
### KNN Classifier
As a preprocessing phase of our procedure, we split the whole dataset into training and test set, respectively with an amount of data of 80-20 percentage. 
For our purposes of classification we trained a KNN classifier by means of a 5-fold Cross Validated Grid Search in the following space of parameters : 
```
parameters = {  
'n_neighbors': [1, 3, 5, 7, 9, 11],  
'weights': ['uniform', 'distance'],  
'p': [1,2]  
}
```
According to accuracy score metric, the best parameters are: 
```
n_neighbors = 11  
weights = uniform  
p = 2
```
### GridSearch Results and Confusion Matrix
|Report| Precision| Recall| F1-score| Support | 
|-------|---------|--------|--------|---------|
|not QRS| 1.00| 1.00| 1.00| 129529|
|QRS | 0.80 | 0.85 | 0.82 | 471 | 
|avg/total| 1.00 | 1.00 | 1.00 | 130000|

|/ |QRS|not QRS|
|-|---|-------|
|QRS|129430 | 99 | 
|not QRS| 72| 399|

# Comparison with an algorithm that does not use a classifier
The algorithm that we took into account is an QRS complex detector, of which you can find further information in the repository linked in the references.  
|Signal|KNN Peaks Detected | Algorithm Peaks Detected | Actual #Peaks | KNN Det. Rate | Algorithm Det.Rate |
|------|-------------------|--------------------------|---------------|---------------|--------------------|
|100|2380|2272| 




# Elapsed time for a single point (patient 100) in seconds
Knn = ![equation](http://latex.codecogs.com/gif.latex?\frac{50}{650000}&space;=&space;7\cdot{10^{-5}})  
No Classifier Algorithm = ![equation](http://latex.codecogs.com/gif.latex?\frac{0.26}{650000}&space;=&space;4\cdot{10^{-6}})

## References 
* [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 
* [QRS complex Detection Algorithm](https://github.com/tru-hy/rpeakdetect/tree/master)


