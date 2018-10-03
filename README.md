# Arrhythmia detection using on standard ECG databases

## Dependencies
The modules are implemented for use with Python 3.x and they consist of the following dependencies:
* scipy
* numpy
* matplotlib

## Introduction
The function of human body is frequently associated with signals of electrical, chemical, or acoustic origin.  
Extracting useful information from these biomedical signals has been found very helpful in explaining and identifying various pathological conditions.  
The most important are the signals which are originated from the heart's electrical activity. This electrical activity of the human heart, though it is quite low in amplitude (about 1 mV) can be detected on the body surface and recorded as an electrocardiogram (ECG) signal.  
The ECG arise because active tissues within the heart generate electrical currents, which flow most intensively within the heart muscle itself, and with lesser intensity throughout the body. The flow of current creates voltages between the sites on the body surface where the electrodes are placed. 

### QRS Region
The QRS complex is the central part of an ECG. It corresponds to the depolarization of the right and left ventricles of the human heart. A Q wave is any downward deflection after the P wave. An R wave follows as an upward deflection, and the S wave is any downward deflection after the R wave. An example of a PQRST segment can be seen in the picture below. 
![QRS complex](https://preview.ibb.co/htTGsn/PQRST.png)  
The normal ECG signal consists of P, QRS and T waves. The QRS interval is a measure of the total duration of ventricular tissue depolarization.  
QRS detection provides the fundamental reference for almost all automated ECG analysis algorithms. Before to perform QRS detection, removal or suppresion of noise is required. 

### MIT-BIH Arrhythmia Database
The MIT-BIH Arrhytmia DB contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects (records 201 and 202 are from the same subject) studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Of these, 23 were chosen at random from a collection of over 4000 Holter tapes, and the other 25 (the "200 series") were selected to include examples of uncommon but clinically important arrhythmias that would not be well represented in a small random sample.  
Each signal contains cardiologists annotations, which describe the behaviour of the signal in the location in which they are placed. 

### Incart Database
This database consists of 75 annotated recordings extracted from 32 Holter records. Each record is 30 minutes long and contains 12 standard leads, each sampled at 257 Hz. Each signal contains cardiologists annotations, which describe the behaviour of the signal in the location in which they are placed. The algorithm generally places beat annotations in the middle of the QRS complex (as determined from all 12 leads). The locations have not been manually corrected, however, and there may be occasional misaligned annotations as a result.

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
|?|Beat not classified during|Non-Beat Annotation | Meaning| 

For the other kind of annotations, which do not refer to QRS region, look at the references.

#### MIT-BIH ANNOTATIONS
|patient|A|N|E|S|V|F|R|j|f|/|a|L|e|J|Q|
|-------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|100|33|2239|0|0|1|0|0|0|0|0|0|0|0|0|0|
|101|3|1860|0|0|0|0|0|0|0|0|0|0|0|0|2|
|102|0|99|0|0|4|0|0|0|56|2028|0|0|0|0|0|
|103|2|2082|0|0|0|0|0|0|0|0|0|0|0|0|0|
|104|0|163|0|0|2|0|0|0|666|1380|0|0|0|0|18|
|105|0|2526|0|0|41|0|0|0|0|0|0|0|0|0|5|
|106|0|1507|0|0|520|0|0|0|0|0|0|0|0|0|0|
|107|0|0|0|0|59|0|0|0|0|2078|0|0|0|0|0|
|108|4|1739|0|0|17|2|0|1|0|0|0|0|0|0|0|
|109|0|0|0|0|38|2|0|0|0|0|0|2492|0|0|0|
|111|0|0|0|0|1|0|0|0|0|0|0|2123|0|0|0|
|112|2|2537|0|0|0|0|0|0|0|0|0|0|0|0|0|
|113|0|1789|0|0|0|0|0|0|0|0|6|0|0|0|0|
|114|10|1820|0|0|43|4|0|0|0|0|0|0|0|2|0|
|115|0|1953|0|0|0|0|0|0|0|0|0|0|0|0|0|
|116|1|2302|0|0|109|0|0|0|0|0|0|0|0|0|0|
|117|1|1534|0|0|0|0|0|0|0|0|0|0|0|0|0|
|118|96|0|0|0|16|0|2166|0|0|0|0|0|0|0|0|
|119|0|1543|0|0|444|0|0|0|0|0|0|0|0|0|0|
|121|1|1861|0|0|1|0|0|0|0|0|0|0|0|0|0|
|122|0|2476|0|0|0|0|0|0|0|0|0|0|0|0|0|
|123|0|1515|0|0|3|0|0|0|0|0|0|0|0|0|0|
|124|2|0|0|0|47|5|1531|5|0|0|0|0|0|29|0|
|200|30|1743|0|0|826|2|0|0|0|0|0|0|0|0|0|
|201|30|1625|0|0|198|2|0|10|0|0|97|0|0|1|0|
|202|36|2061|0|0|19|1|0|0|0|0|19|0|0|0|0|
|203|0|2529|0|0|444|1|0|0|0|0|2|0|0|0|4|
|205|3|2571|0|0|71|11|0|0|0|0|0|0|0|0|0|
|207|107|0|105|0|105|0|86|0|0|0|0|1457|0|0|0|
|208|0|1586|0|2|992|373|0|0|0|0|0|0|0|0|2|
|209|383|2621|0|0|1|0|0|0|0|0|0|0|0|0|0|
|210|0|2423|1|0|194|10|0|0|0|0|22|0|0|0|0|
|212|0|923|0|0|0|0|1825|0|0|0|0|0|0|0|0|
|213|0|3251|0|0|0|0|0|0|0|0|0|0|0|0|0|
|214|0|0|0|0|256|1|0|0|0|0|0|2003|0|0|2|
|215|3|3195|0|0|164|1|0|0|0|0|0|0|0|0|0|
|217|0|244|0|0|162|0|0|0|260|1542|0|0|0|0|0|
|219|7|2082|0|0|64|1|0|0|0|0|0|0|0|0|0|
|220|94|1954|0|0|0|0|0|0|0|0|0|0|0|0|0|
|221|0|2031|0|0|396|0|0|0|0|0|0|0|0|0|0|
|222|208|2062|0|0|0|0|0|212|0|0|0|0|0|1|0|
|223|72|2029|0|0|473|14|0|0|0|0|1|0|16|0|0|
|228|3|1688|0|0|362|0|0|0|0|0|0|0|0|0|0|
|230|0|2255|0|0|1|0|0|0|0|0|0|0|0|0|0|
|231|1|314|0|0|2|0|1254|0|0|0|0|0|0|0|0|
|232|1382|0|0|0|0|0|397|1|0|0|0|0|0|0|0|
|233|7|2230|0|0|831|11|0|0|0|0|0|0|0|0|0|
|234|0|2700|0|0|3|0|0|0|0|0|0|0|0|50|0|

### QRS and R peak detection
The first phase we have to work on is to describe different approaches for encountering the problems of QRS and R peak detection, and evaluating and comparing the results obtained using the same standard ECG databases.  
In order to do this, we used three different algorithms: 
- one based on the KNN classifier
- one based on a heuristic method
- one based on the PanTompkins approach

```
pip install wfdb
```
# KNN APPROACH
## Signal Preprocessing
Before the classification phase, signals are processed by using a band pass filter, in order to reduce the recording noise that would lead to uncorrect classifications.  
The sample frequency is set to 360 sample per second.  
We used the butter method from scipy to obtain the filter coefficients in the following way: 
```
b, a = signal.butter(N, Wn, btype="bandpass")
```
where N is the order of the filter, Wn is a scalar giving the critical frequencies, btype is the type of the filter.  
Once we get the coefficients, we apply a digital filter forward and backward to a signal :
```
filtered_channel = filtfilt(b, a, x)
```
where b and a are the coefficients we computed above and x is the array of data to be filtered.  
Once we filtered the channels, we apply the gradient to the whole signal with the diff function from numpy. It actually calculates the n-th discrete difference along the given axis. After this, we just square the signal to get no zeros and eventually, we normalize the gradient.


## Data Loading 
Data are available in the PhysioNet website, precisely at the link below:  
https://www.physionet.org/physiobank/database/mitdb/  
Raw signals are loaded inside the Python module using the wfdb library.

## Classification
In this scope, the QRS detection problem is encountered as a binary classification problem:  
A signal is decomposed in features of variable size that can be detected either as a QRS complex or not, and considered as input for a KNN classifier.  
A portion of the signal is labeled as a QRS complex depending on whether it contains a Beat Annotation.  
Then a classifier is trained with the 80% of the length of the signal, while the last 20% is used for testing purpose.  
KNN classifiers are trained by means of a 5-fold Cross Validated Grid Search in the following space of parameters : 
```
parameters = {  
'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],  
'weights': ['uniform', 'distance'] 
}
```
The Grid Search provides, for eache signal, the best configuration of parameters according to the accuracy score.  
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html for further readings.

## Feature Extraction
In this approach, the signal is divided in windows of samples of fixed sizes of 36 samples, which are considered as features. The feature vector is composed by the gradients referred to all samples in the window.  
A window is labeled as QRS region if contains a Beat Annotation.

# RPEAK DETECTION HEURISTIC

## Signal Processing
Also here, the filter chosen is a passband since they maximize the energy of different QRS complexes and reduce the effect of P/T waves, motion artifacts and muscle noise. After filtering, a first-order forward differentiation is applied to emphasize large slope and high-frequency content of the QRS complex. The derivative operation reduces the effect of large P/T waves. A rectification process is then employed to obtain a positive-valued signal that eliminates detection problems in case of negative QRS complexes. In this approach, a new nonlinear transformation based on squaring, thresholding process and Shannon energy transformation is designed to avoid to misconsider some R-peak. 

For further information, please see the references. 

# Pan Tompkins Approach

In order to get all the peaks of each signal we used a matlab version of the algorithm wrote by Pan-Tompkins which you can find in the repository. It takes the signals as input and give a list containing the peaks' location as output.

# EVALUATION
## MIT-BIH Arrythmia Database
|         | KNN       |        | generic   |        |
|---------|-----------|--------|-----------|--------|
| signal  | precision | recall | precision | recall |
| 100     | 1         | 0,962  | 0,557     | 0,999  |
| 101     | 0,997     | 0,965  | 0,949     | 0,989  |
| 103     | 1         | 0,963  | 0,995     | 0,99   |
| 105     | 0,976     | 0,952  | 0,218     | 0,713  |
| 106     | 0,998     | 0,934  | 0,859     | 0,883  |
| 108     | 0,881     | 0,772  | 0,065     | 0,437  |
| 109     | 0,998     | 0,962  | 0,947     | 0,976  |
| 111     | 0,999     | 0,91   | 0,89      | 0,836  |
| 112     | 1         | 0,968  | 0,978     | 0,462  |
| 113     | 1         | 0,968  | 1         | 0,832  |
| 114     | 1         | 0,97   | 0,996     | 0,981  |
| 115     | 1         | 0,942  | 1         | 0,95   |
| 116     | 0,999     | 0,824  | 0,877     | 0,879  |
| 117     | 1         | 0,961  | 0,995     | 0,48   |
| 118     | 0,999     | 0,962  | 0,547     | 0,293  |
| 119     | 0,996     | 0,96   | 0,998     | 0,325  |
| 121     | 1         | 0,967  | 0,009     | 0,101  |
| 122     | 1         | 0,967  | 1         | 0,224  |
| 123     | 1         | 0,968  | 0,998     | 0,905  |
| 124     | 1         | 0,948  | 0,836     | 0,644  |
| 200     | 0,999     | 0,956  | 0,715     | 0,959  |
| 201     | 0,997     | 0,885  | 0,881     | 0,978  |
| 202     | 1         | 0,965  | 0,951     | 0,987  |
| 203     | 0,989     | 0,735  | 0,271     | 0,804  |
| 205     | 0,998     | 0,96   | 0,765     | 1      |
| 207     | 0,977     | 0,683  | 0,604     | 0,964  |
| 208     | 0,932     | 0,8    | 0,836     | 0,958  |
| 209     | 0,999     | 0,964  | 0,999     | 0,977  |
| 210     | 0,995     | 0,916  | 0,085     | 0,97   |
| 212     | 1         | 0,957  | 0,969     | 0,958  |
| 213     | 0,993     | 0,95   | 0,989     | 0,857  |
| 214     | 0,999     | 0,957  | 0,944     | 0,887  |
| 215     | 1         | 0,966  | 0,867     | 0,996  |
| 219     | 1         | 0,951  | 0,992     | 0,396  |
| 220     | 1         | 0,963  | 1         | 0,844  |
| 221     | 1         | 0,956  | 0,995     | 0,911  |
| 222     | 0,949     | 0,922  | 0,638     | 0,857  |
| 223     | 1         | 0,851  | 0,852     | 0,929  |
| 228     | 0,995     | 0,946  | 0,193     | 0,705  |
| 230     | 1         | 0,949  | 0,992     | 0,992  |
| 231     | 1         | 0,957  | 1         | 0,998  |
| 232     | 0,999     | 0,949  | 0,988     | 0,948  |
| 233     | 0,996     | 0,945  | 0,986     | 0,939  |
| 234     | 1         | 0,942  | 0,988     | 0,969  |
| average | 0,986     | 0,919  | 0,789     | 0,788  |




# RR ANALYSIS FOR BEAT CLASSIFICATION
Going on we have to classify the peaks we located with the approaches described above. For this purpose we decided to use a rule based approach described by Tsipouras. The paper we took into account can be found in the references at the end of this readme. Once we labeled the beats, we went on labeling the RR intervals and the results can be found below. Following the rules written in the paper we can finally detect Arrhythmias.

# RR RESULTS
## SENSITIVITY
### 100 series
|-|N|PVC|VF|BII|
|-|-|---|--|---|
|RPEAK|98%|62%|-|-|
|ANNOTATION|99%|85%|-|-|
|PAN-TOMPKINS|90%|41%|-|-|

### 200 series
|-|N|PVC|VF|BII|
|-|-|---|--|---|
|RPEAK|88%|35%|1%|100%|
|ANNOTATION|93%|74%|51%|100%|
|PAN-TOMPKINS|87%|37%|36%|100%|

## PRECISION
 ### 100 series
|-|N|PVC|VF|BII|
|-|-|---|--|---|
|RPEAK|99%|34%|-|0%|
|ANNOTATION|99%|46%|-|-|
|PAN-TOMPKINS|98%|20%|0%|0%|
### 200 series

|-|N|PVC|VF|BII|
|-|-|---|--|---|
|RPEAK|92%|26%|25%|0.2%|
|ANNOTATION|98%|55%|50%|0.25%|
|PAN-TOMPKINS|92%|25%|40%|0.2%|

## Beat Classification
### Weighted SVM


|-|N|S|V|F|Average|
|-|-|-|-|-|-|
|Recall|0.795|0.688|0.829|0.928|0.81|
|Precision|0.985|0.318|0.878|0.05|0.55|

Average Accuracy: 0.795
rebalanced dataset, with scale factors [-14, 4, 1, 10]
C = 0.1
# Neural Network Approach


## References 
* 1) [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* 2) [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* 3) [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* 4) [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 
* 5) [QRS complex Detection Algorithm](https://github.com/tru-hy/rpeakdetect/tree/master)
* 6) [Arrhthmya Classification](https://www.sciencedirect.com/science/article/pii/S0933365704000806)


