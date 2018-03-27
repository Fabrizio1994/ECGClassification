# QRS detection using K-Nearest Neighbor algorithm (KNN) and evaluation on standard ECG databases

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

## MIT-BIH Arrhythmia Database
The MIT-BIH Arrhytmia DB contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects (records 201 and 202 are from the same subject) studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Of these, 23 were chosen at random from a collection of over 4000 Holter tapes, and the other 25 (the "200 series") were selected to include examples of uncommon but clinically important arrhythmias that would not be well represented in a small random sample.  
Each signal contains cardiologists annotations, which describe the behaviour of the signal in the location in which they are placed. In particular, the Beat Annotations classify each QRS complex with one of the following labels:

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

## QRS and R peak detection
The aim of this work is to describe different approaches for encountering the problems of QRS and R peak detection, and evaluating and comparing the results obtained using the same standard ECG databases.  
In order to do this, we used two different algorithms: one based on the KNN classifier and the other based on a heuristic method.

## Data Loading 
Data are available in the PhysioNet website, precisely at the link below:  
https://www.physionet.org/physiobank/database/mitdb/  
Raw signals are loaded inside the Python module using the wfdb library.

```
pip install wfdb
```
## KNN APPROACHES
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


## Classification
In this scope, the QRS detection problem is encountered as a binary classification problem:  
A signal is decomposed in features of variable size that can be detected either as a QRS complex or not, and considered as input for a KNN classifier.  
A portion of the signal is labeled as a QRS complex depending on whether it contains a Beat Annotation.  
Then a classifier is trained with the 80% of the length of the signal, while the last 20% is used for testing purpose.  
KNN classifiers are trained by means of a 5-fold Cross Validated Grid Search in the following space of parameters : 
```
parameters = {  
'n_neighbors': [1, 3, 5, 7, 9, 11],  
'weights': ['uniform', 'distance'],  
'p': [1,2]  
}
```
The Grid Search provides, for eache signal, the best configuration of parameters according to the accuracy score.  
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html for further readings.



### Feature Extraction
We executed the above procedure and stored the results obtained defining features in two different ways: at sample level and at window level.
#### Sample level
As suggested in the paper [QRS detection using KNN], each feature is a 2D vector containing the gradient values for a given sample, one for each channel. The feature matrix shape is then 650.000 X 2.  
Each feature is labeled with 1 or -1 whether it is located in a range, called window, of variable size around a Beat Annotation. We stored the results obtained with windows of size 10, 20 and 50 samples.

#### Window level
In this approach, the signal is divided in windows of samples of fixed sizes (again 10, 20 and 50 samples), which are considered as features. The feature vector is composed by the gradients referred to all samples in the window.  
A window is labeled as QRS region if contains a Beat Annotation.
## RPEAK DETECTION HEURISTIC

### Signal Processing
Also here, the filter chosen is a passband since they maximize the energy of different QRS complexes and reduce the effect of P/T waves, motion artifacts and muscle noise. After filtering, a first-order forward differentiation is applied to emphasize large slope and high-frequency content of the QRS complex. The derivative operation reduces the effect of large P/T waves. A rectification process is then employed to obtain a positive-valued signal that eliminates detection problems in case of negative QRS complexes. In this approach, a new nonlinear transformation based on squaring, thresholding process and Shannon energy transformation is designed to avoid to misconsider some R-peak. 

For further information, please see the reference [n°5]. 


### Evaluation

#### Window Level 2 Channels
|SIGNAL|SE_10|SE_20|SE_50|RPEAK-SE|DIFF|
|-|-|-|-|-|-|
|100|99.041|99.332|99.129|99.563|0.037|
|101|97.222|98.67|99.128|100.0|0.0|
|102|94.118|94.432|97.882|93.593|0.386|
|103|99.231|98.423|98.778|100.0|0.0|
|104|95.117|94.471|98.693|97.753|0.124|
|105|94.499|96.238|99.43|98.485|0.087|
|106|88.915|93.97|96.296|85.854|0.182|
|107|95.422|97.789|97.393|54.545|3.359|
|108|82.095|84.225|92.398|43.386|1.287|
|109|95.714|97.287|98.077|97.804|0.224|
|111|96.061|98.485|98.481|97.025|0.255|
|112|98.156|98.394|98.403|42.339|3.667|
|113|99.716|99.104|99.204|99.731|0.0|
|114|97.436|98.254|98.638|94.588|1.299|
|115|98.507|99.483|100.0|99.742|0.078|
|116|97.737|99.596|98.542|99.599|0.533|
|117|90.26|96.98|98.107|99.032|2.775|
|118|97.327|99.553|99.107|92.357|2.692|
|119|99.261|99.495|99.744|99.747|0.018|
|121|96.802|98.232|99.444|42.118|1.05|
|122|97.6|99.22|99.797|100.0|0.103|
|123|98.955|99.701|99.67|100.0|0.259|
|124|96.53|98.671|99.029|99.714|0.014|
|200|92.593|95.602|97.037|97.963|0.439|
|201|95.98|96.306|97.8|96.026|0.08|
|202|97.356|98.259|98.578|99.825|0.039|
|203|85.275|91.181|96.239|86.092|0.352|
|205|97.605|98.864|99.282|98.0|0.098|
|207|92.011|91.369|96.533|81.21|1.365|
|208|92.308|95.269|97.35|70.848|0.155|
|209|97.329|98.635|99.317|100.0|0.051|
|210|96.296|96.196|97.688|95.238|0.058|
|212|99.257|98.947|99.29|99.81|0.053|
|213|96.148|97.118|98.438|99.058|0.035|
|214|97.577|97.059|98.698|99.342|0.053|
|215|98.426|99.096|99.559|97.301|0.063|
|217|93.096|97.183|96.889|42.691|3.565|
|219|99.302|98.109|99.314|100.0|0.002|
|220|99.281|99.239|99.757|100.0|0.0|
|221|97.951|97.921|99.387|99.566|0.011|
|222|97.125|97.053|99.398|98.942|0.453|
|223|94.559|97.243|98.643|94.423|0.193|
|228|93.187|96.737|98.313|88.702|0.407|
|230|98.113|98.416|99.576|96.933|0.017|
|231|99.363|98.75|99.342|100.0|0.0|
|232|97.581|99.722|99.202|99.449|0.166|
|233|90.645|94.272|97.293|84.426|0.373|
|234|99.437|98.674|100.0|100.0|0.011|




#### Sample Level 2 Channels
##### 10 sample window
|SIGNAL|TP|TN|FP|FN|DER|SE|
|-|-|-|-|-|-|-|
|124|13618|112275|1228|2879|0.30158613599647527|82.5483421228102|
|111|17469|106444|2035|4052|0.3484458183067147|81.17187863017517|
|112|19809|102371|1733|6087|0.39477005401585136|76.49443929564411|
|231|14648|113293|768|1291|0.14056526488257784|91.90037016123974|
|222|16437|101770|3091|8702|0.7174666910020077|65.38446238911652|
|109|23010|102805|1597|2588|0.181877444589309|89.88983514337058|
|214|18118|105339|1542|5001|0.36113257533944143|78.36844154158918|
|230|20691|105495|1557|2257|0.18433135179546664|90.1647202370577|
|122|23745|104554|556|1145|0.07163613392293114|95.39975893933307|
|210|16545|100359|2859|10237|0.7915382290722273|61.77656635053395|
|215|32620|93402|2096|1882|0.12194972409564685|94.54524375398528|
|101|11670|109513|1356|7461|0.7555269922879178|61.00047044064607|
|223|25090|102060|1401|1449|0.11359107214029494|94.54011078036098|
|228|19570|107456|1551|1423|0.15196729688298416|93.22155004048969|
|220|20597|108863|283|257|0.026217410302471232|98.76762251846168|
|123|14404|113800|739|1057|0.1246875867814496|93.16344350300757|
|209|25548|96521|2856|5075|0.3104352591200877|83.42748914214806|
|217|17234|104589|2903|5274|0.47446907276314265|76.56833125999644|

##### 20 sample window
|SIGNAL|TP|TN|FP|FN|DER|SE|
|-|-|-|-|-|-|-|
|124|5050|122365|930|1655|0.5118811881188119|75.31692766592096|
|111|8449|120587|461|503|0.1140963427624571|94.38114387846291|
|112|8710|118521|714|2055|0.31791044776119404|80.91035764050163|
|231|6105|123164|261|470|0.11973791973791974|92.85171102661597|
|222|9399|118403|1067|1131|0.2338546653899351|89.25925925925927|
|109|8041|117266|2023|2670|0.5836338763835344|75.07235552236018|
|214|7079|118958|1410|2553|0.559824834016104|73.49460132890366|
|230|8507|119835|696|962|0.1948983190313859|89.84053226317457|
|122|10148|119100|402|350|0.07410327158060702|96.66603162507144|
|210|7461|116982|1848|3709|0.7448063262297279|66.79498657117279|
|215|13426|114986|986|602|0.1182779681215552|95.70858283433134|
|101|6009|121485|691|1815|0.4170411050091529|76.8021472392638|
|223|10130|117994|897|979|0.18519249753208292|91.18732559186246|
|228|7342|120717|712|1229|0.26436938163988016|85.66094971415238|
|220|8436|121271|144|149|0.0347321005215742|98.26441467676179|
|123|6116|123411|240|233|0.07733812949640288|96.3301307292487|
|209|11976|116737|666|621|0.10746492985971944|95.0702548225768|
|217|6687|118958|1768|2587|0.6512636458800658|72.10480914384301|


##### 50 sample window
|SIGNAL|TP|TN|FP|FN|DER|SE|
|-|-|-|-|-|-|-|
|124|13618|112275|1228|2879|0.30158613599647527|82.5483421228102|
|111|17469|106444|2035|4052|0.3484458183067147|81.17187863017517|
|112|19809|102371|1733|6087|0.39477005401585136|76.49443929564411|
|231|14648|113293|768|1291|0.14056526488257784|91.90037016123974|
|222|16437|101770|3091|8702|0.7174666910020077|65.38446238911652|
|109|23010|102805|1597|2588|0.181877444589309|89.88983514337058|
|214|18118|105339|1542|5001|0.36113257533944143|78.36844154158918|
|230|20691|105495|1557|2257|0.18433135179546664|90.1647202370577|
|122|23745|104554|556|1145|0.07163613392293114|95.39975893933307|
|210|16545|100359|2859|10237|0.7915382290722273|61.77656635053395|
|215|32620|93402|2096|1882|0.12194972409564685|94.54524375398528|
|101|11670|109513|1356|7461|0.7555269922879178|61.00047044064607|
|223|25090|102060|1401|1449|0.11359107214029494|94.54011078036098|
|228|19570|107456|1551|1423|0.15196729688298416|93.22155004048969|
|220|20597|108863|283|257|0.026217410302471232|98.76762251846168|
|123|14404|113800|739|1057|0.1246875867814496|93.16344350300757|
|209|25548|96521|2856|5075|0.3104352591200877|83.42748914214806|
|217|17234|104589|2903|5274|0.47446907276314265|76.56833125999644|


#### Window Level One Channel

|SIGNAL|KNN-SE-10|KNN-SE-20|KNN-SE-50|RPEAK-SE|DIFF|
|------|---------|---------|---------|--------|----|
|100|99.099|98.878|99.350|99.563|0.037|
|101|96.954|97.010|99.204|100.0|0.0|
|102|96.321|97.278|96.336|93.593|0.386|
|103|98.750|99.522|100.0|100.0|0.0|
|104|94.965|95.955|96.798|97.753|0.124|
|105|95.372|97.348|98.594|98.485|0.087|
|106|88.177|93.478|96.741|85.854|0.182|
|107|94.562|97.065|98.198|54.545|3.359|
|108|67.605|82.080|88.823|43.386|1.287|
|109|91.717|94.683|97.868|97.804|0.224|
|111|89.021|93.954|97.228|97.025|0.255|
|112|79.393|84.551|93.142|42.339|3.667|
|113|100.0|98.691|99.705|99.731|0.0|
|114|81.413|91.596|96.113|94.588|1.299|
|115|98.441|99.496|99.212|99.742|0.078|
|116|95.726|97.357|97.613|99.599|0.533|
|117|93.092|96.551|99.071|99.032|2.775|
|118|90.762|94.235|98.013|92.357|2.692|
|119|98.232|99.25|100.0|99.747|0.018|
|121|42.894|80.474|89.637|42.118|1.05|
|122|95.801|97.125|99.198|100.0|0.103|
|123|98.032|98.0|99.044|100.0|0.259|
|124|95.195|96.261|97.264|99.714|0.014|
|200|89.962|94.545|97.560|97.963|0.439|
|201|90.957|95.238|97.761|96.026|0.08|
|202|97.374|97.5|99.321|99.825|0.039|
|203|82.587|89.123|95.084|86.092|0.352|
|205|98.076|98.175|99.812|98.0|0.098|
|207|84.254|91.825|95.142|81.21|1.365|
|208|92.617|95.076|97.391|70.848|0.155|
|209|97.689|98.029|99.134|100.0|0.051|
|210|91.344|94.776|98.720|95.238|0.058|
|212|97.868|99.156|99.458|99.81|0.053|
|213|96.810|98.897|99.526|99.058|0.035|
|214|94.570|98.0|98.156|99.342|0.053|
|215|96.947|98.033|99.689|97.301|0.063|
|217|92.452|95.512|97.058|42.691|3.565|
|219|95.312|98.095|99.530|100.0|0.002|
|220|98.218|99.529|99.750|100.0|0.0|
|221|96.529|98.709|99.132|99.566|0.011|
|222|91.188|93.939|94.939|98.942|0.453|
|223|88.909|93.385|96.577|94.423|0.193|
|228|90.886|92.462|96.296|88.702|0.407|
|230|98.698|99.120|99.349|96.933|0.017|
|231|99.053|98.165|99.698|100.0|0.0|
|232|91.428|97.109|99.157|99.449|0.166|
|233|87.055|93.421|96.173|84.426|0.373|
|234|98.059|99.815|99.625|100.0|0.011|


# Elapsed time for a single point (patient 100) in seconds
Knn = ![equation](http://latex.codecogs.com/gif.latex?\frac{50}{650000}&space;=&space;7\cdot{10^{-5}})  
RPeakDetector = ![equation](http://latex.codecogs.com/gif.latex?\frac{0.26}{650000}&space;=&space;4\cdot{10^{-6}})

## References 
* 1) [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* 2) [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* 3) [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* 4) [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 
* 5) [QRS complex Detection Algorithm](https://github.com/tru-hy/rpeakdetect/tree/master)



