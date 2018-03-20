# QRS detection using K-Nearest Neighbor algorithm (KNN) and evaluation on standard ECG databases
## Introduction
The function of human body is frequently associated with signals of electrical, chemical, or acoustic origin. Extracting useful information from these biomedical signals has been found very helpful in explaining and identifying various pathological conditions. The most important are the signals which are originated from the heart's electrical activity. This electrical activity of the human heart, though it is quite low in amplitude (about 1 mV) can be detected on the body surface and recorded as an electrocardiogram (ECG) signal. The ECG arise because active tissues within the heart generate electrical currents, which flow most intensively within the heart muscle itself, and with lesser intensity throughout the body. The flow of current creates voltages between the sites on the body surface where the electrodes are placed. The normal ECG signal consists of P, QRS and T waves. The QRS interval is a measure of the total duration of ventricular tissue depolarization. QRS detection provides the fundamental reference for almost all automated ECG analysis algorithms. Before to perform QRS detection, removal or suppresion of noise is required. The aim of this work is to explore the merits of KNN algorithm as an ECG delineator. The KNN method is an instance based learning method that stores all available data points and classifies new data points based on similarity measure. In KNN, the each training data consists of a set of vectors and every vector has its own positive or negative class label, where K represents the number of neighbors. 

### Dependencies
The modules are implemented for use with Python 3.x and they consist of the following dependencies:
* scipy
* numpy
* matplotlib
```
pip install wfdb
```

### Repository directory structure 
```
├── Main.py
|
├── Evaluation.py
|
├── KNN.py
|
├── RPeakDetection.py
|
├── Utility.py
|
├── FeatureExtraction.py
|
├── GridSearch.py
|
├── License.md
|
└── README.md
```

# MIT-BIH Arrhythmia Database
The MIT-BIH Arrhytmia DB contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects (records 201 and 202 are from the same subject) studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Of these, 23 were chosen at random from a collection of over 4000 Holter tapes, and the other 25 (the "200 series") were selected to include examples of uncommon but clinically important arrhythmias that would not be well represented in a small random sample. Approximately 60% of the subjects were inpatients. The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range. The digitization rate was chosen to accommodate the use of simple digital notch filters to remove 60 Hz (mains frequency) interference. Six of the 48 records contain a total of 33 beats that remain unclassified, because the cardiologist-annotators were unable to reach agreement on the beat types. The annotators were instructed to use all evidence available from both signals to identify every detectable QRS complex. The database contains seven episodes of loss of signal or noise so severe in both channels simultaneously that QRS complexes cannot be detected; these episodes are all quite short and have a total duration of 10s. In all of the remaining data, every QRS complex was annotated, about 109.000 in all. 

## QRS Region
The QRS complex is the central part of an ECG. It corresponds to the depolarization of the right and left ventricles of the human heart. A Q wave is any downward deflection after the P wave. An R wave follows as an upward deflection, and the S wave is any downward deflection after the R wave. Since each annotation we got from the MIT-BIH Database is not precisely associated to an R-Peak but to a whole QRS complex, we had to consider a window of fixed size (±5, ±10, ±25) with the R-peak as center point, considering as good annotations those ones that will lay inside the window. An example of a PQRST segment can be seen in the picture below. 

![QRS complex](https://preview.ibb.co/htTGsn/PQRST.png)

For further descriptions, please see the references. 

## Annotations 
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


# The Algorithms
The aim of this work is to provide two efficient solutions for the problems of QRS detection and, more precisely, R-Peak detection. 
In order to do this, we used two different algorithms: one based on the KNN classifier and the other based on a heuristic method.
Every signal is annotated by cardiologists with the locations of the QRS complexes, and labeled with a symbol from the list above. In this scope, the QRS detection problem is encountered as a binary classification problem:
each sample point of a given signal is considered as a feature that can be classified either as a QRS complex or not.
Each signal is considered for 80% of its length as training data and the rest as test data.
Both algorithms are actually divided in four main phases: Data loading, Signal processing, Feature Extraction and Evaluation.

## KNN Approach

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
The sample frequency is set to 360 sample per second. 
We used the butter method from scipy to obtain the filter coefficients in the following way: 
```
b, a = signal.butter(N, Wn, btype)
```
where N is the order of the filter, Wn is a scalar giving the critical frequencies, btype is the type of the filter.
Once we get the coefficients, we apply a digital filter forward and backward to a signal :
```
filtered_channel = filtfilt(b, a, x)
```
where b and a are the coefficients we computed above and x is the array of data to be filtered.
Once we filtered the channels, we apply the gradient to the whole signal with the diff function from numpy. It actually calculates the n-th discrete difference along the given axis. After this, we just square the signal to get no zeros and eventually, we normalize the gradient in the following way:


### Feature Extraction


### Evaluation


### Results

|SIGNAL|SE-10|SE-20|SE-50|SE-50-RPEAK|DIFF-RPEAK|
|-|-|-|-|-|-|
|124|96.52996845425868|98.67109634551495|99.02912621359224|100.0|5.354285714285714|
|111|96.06126914660832|98.48484848484848|98.48101265822785|86.95652173913044|7.25|
|112|98.15573770491804|98.39357429718876|98.40319361277446|97.98387096774194|14.965020576131687|
|231|99.36305732484077|98.75|99.3421052631579|100.0|0.0|
|222|97.1252566735113|97.05263157894737|99.3975903614458|97.88359788359789|2.4306306306306307|
|109|95.71428571428572|97.28682170542635|98.07692307692307|98.40319361277446|0.39148073022312374|
|214|97.57709251101322|97.05882352941177|98.69848156182212|95.83333333333334|0.10983981693363844|
|230|98.11320754716981|98.41628959276018|99.57627118644068|85.88957055214725|4.402380952380953|
|122|97.6|99.22027290448344|99.79716024340772|100.0|17.557344064386317|
|210|96.29629629629629|96.19565217391305|97.6878612716763|95.80952380952381|0.7952286282306164|
|215|98.42632331902719|99.09638554216868|99.55947136563876|86.80659670164917|16.33678756476684|
|101|97.22222222222221|98.67021276595744|99.12790697674419|100.0|0.0|
|223|94.55909943714822|97.24264705882352|98.64341085271317|91.73076923076923|1.8029350104821802|
|228|93.18734793187348|96.73659673659674|98.3132530120482|96.39423076923077|0.46633416458852867|
|220|99.28057553956835|99.23857868020305|99.7566909975669|100.0|20.44951923076923|
|123|98.95470383275261|99.7005988023952|99.66996699669967|100.0|22.158576051779935|
|209|97.32888146911519|98.63481228668942|99.31740614334471|100.0|2.823327615780446|
|217|93.0957683741648|97.1830985915493|96.88888888888889|11.600928074245939|0.28|






## R-Peak Detection Heuristic

### Reading Data
This phase is executed in the same way as we did for the KNN approach.

### Signal Processing
Also here, the filter chosen is a passband since they maximize the energy of different QRS complexes and reduce the effect of P/T waves, motion artifacts and muscle noise. After filtering, a first-order forward differentiation is applied to emphasize large slope and high-frequency content of the QRS complex. The derivative operation reduces the effect of large P/T waves. A rectification process is then employed to obtain a positive-valued signal that eliminates detection problems in case of negative QRS complexes. In this approach, a new nonlinear transformation based on squaring, thresholding process and Shannon energy transformation is designed to avoid to misconsider some R-peak. 

For further information, please see the reference [n°5]. 

### Feature Extraction

### Evaluation

# Results
|SIGNAL|TP|TN|FP|FN|DER|SE|
|-|-|-|-|-|-|-|
|100|456|129541|1|2|0.006|99.56|
|101|362|129638|0|0|0.0|100.0|
|102|409|129535|28|28|0.136|93.59|
|103|407|129593|0|0|0.0|100.0|
|104|435|129545|10|10|0.0459|97.75|
|105|520|129462|10|8|0.0346|98.48|
|106|352|129539|51|58|0.309|85.85|
|107|234|129376|195|195|1.666|54.54|
|108|164|129410|212|214|2.597|43.38|
|109|490|129488|11|11|0.044|97.80|
|111|424|129550|13|13|0.061|97.02|
|112|210|129218|286|286|2.723|42.33|
|113|371|129628|0|1|0.002|99.73|
|114|402|129552|23|23|0.114|94.58|
|115|386|129613|0|1|0.002|99.74|
|116|497|129500|1|2|0.006|99.59|
|117|307|129687|3|3|0.019|99.03|
|118|435|129493|36|36|0.165|92.35|
|119|395|129603|1|1|0.005|99.74|
|121|179|129330|245|246|2.743|42.11|
|122|497|129503|0|0|0.0|100.0|
|123|309|129691|0|0|0.0|100.0|
|124|349|129649|1|1|0.005|99.71|
|200|481|129500|9|10|0.039|97.96|
|201|435|129534|13|18|0.071|96.02|
|202|570|129429|0|1|0.001|99.82|
|203|489|129380|52|79|0.267|86.09|
|205|490|129492|8|10|0.036|98.0|
|207|255|129513|173|59|0.909|81.21|
|208|401|129270|164|165|0.820|70.84|
|209|583|129417|0|0|0.0|100.0|
|210|500|129470|5|25|0.06|95.23|
|212|524|129474|1|1|0.003|99.80|
|213|631|129358|5|6|0.017|99.05|
|214|453|129541|3|3|0.013|99.34|
|215|649|129317|16|18|0.052|97.30|
|217|184|129322|247|247|2.684|42.69|
|219|451|129549|0|0|0.0|100.0|
|220|416|129584|0|0|0.0|100.0|
|221|459|129538|1|2|0.006|99.56|
|222|561|129430|3|6|0.016|98.94|
|223|491|129452|28|29|0.116|94.42|
|228|369|129533|51|47|0.265|88.70|
|230|474|129496|15|15|0.063|96.93|
|231|371|129629|0|0|0.0|100.0|
|232|361|129635|2|2|0.011|99.44|
|233|515|129296|94|95|0.366|84.42|
|234|540|129460|0|0|0.0|100.0|



# READ BEFORE DELETE
## READ BEFORE DELETE
### READ BEFORE DELETE

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
* 1) [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* 2) [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* 3) [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* 4) [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 
* 5) [QRS complex Detection Algorithm](https://github.com/tru-hy/rpeakdetect/tree/master)


