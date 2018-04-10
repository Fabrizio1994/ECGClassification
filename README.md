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


## EVALUATION
### MIT-BIH Arrythmia Database

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

#### Window Level One Channel
|patient|SE-10|SE-20|SE-30|SE-50|SE-70|SE-90|SE-110|SE-130|SE-150|
|-------|-----|-----|-----|-----|-----|-----|------|------|------|
|100|99.084|99.558|99.560|99.777|99.773|99.772|98.497|99.786|100.0|
|101|98.461|98.622|99.420|99.465|99.466|100.0|99.737|99.473|99.720|
|102|94.618|93.763|98.387|97.380|97.619|98.861|99.022|98.800|98.113|
|103|99.313|99.770|100.0|100.0|99.533|100.0|99.753|99.754|100.0|
|104|90.315|92.763|96.559|97.186|98.672|98.0|98.313|99.150|99.109|
|105|96.887|99.168|97.402|98.832|99.401|98.825|98.073|99.029|99.408|
|106|92.802|93.750|95.903|98.708|98.571|97.963|98.473|99.765|97.815|
|107|94.209|96.321|97.362|96.559|97.948|98.792|98.130|98.122|99.772|
|108|75.644|84.153|90.935|90.084|93.854|93.678|95.873|97.843|96.042|
|109|97.302|98.073|98.661|99.445|99.418|99.032|99.802|99.175|98.861|
|111|94.318|97.228|98.317|97.995|99.097|98.004|98.740|98.823|98.568|
|112|97.064|99.195|99.799|99.796|99.417|99.588|99.589|99.590|100.0|
|113|99.452|99.468|100.0|99.737|99.456|100.0|100.0|99.729|100.0|
|114|79.057|93.281|97.567|98.365|98.235|98.918|98.955|99.193|97.395|
|115|99.507|100.0|99.506|99.748|100.0|100.0|99.742|99.740|100.0|
|116|96.645|98.443|98.308|97.251|98.159|98.336|99.152|98.767|99.575|
|117|90.759|95.268|97.359|98.407|99.307|98.586|98.630|99.674|98.958|
|118|98.064|99.555|98.257|99.134|99.360|100.0|98.886|99.331|99.760|
|119|99.758|99.514|99.496|100.0|99.501|99.763|99.757|99.752|100.0|
|121|98.449|98.987|99.479|99.736|98.866|99.193|98.898|100.0|99.737|
|122|98.643|99.433|100.0|99.595|100.0|99.4|100.0|99.792|98.977|
|123|100.0|99.356|99.326|99.335|99.664|99.658|100.0|99.669|99.682|
|124|97.345|96.820|98.493|99.127|99.322|97.476|99.137|98.795|100.0|
|200|94.509|97.683|97.445|97.238|98.510|98.731|99.233|99.220|99.622|
|201|94.414|97.692|96.649|97.5|97.229|98.561|98.543|98.300|98.910|
|202|99.218|98.727|99.115|99.533|98.831|99.540|99.313|99.289|98.853|
|203|84.013|92.218|95.559|95.197|95.476|95.950|97.063|97.319|97.631|
|205|98.568|99.190|99.252|100.0|99.624|99.812|99.809|100.0|99.429|
|207|85.0|90.833|92.288|93.783|95.077|94.751|96.348|97.340|96.476|
|208|88.215|90.398|93.965|97.213|98.363|98.657|98.415|98.046|97.689|
|209|97.800|99.205|99.658|99.829|99.841|99.670|100.0|99.483|100.0|
|210|94.609|95.0|98.689|99.436|99.243|99.616|99.239|99.111|99.021|
|212|98.943|99.306|99.813|99.615|99.467|100.0|100.0|100.0|99.820|
|213|94.391|98.430|98.496|98.816|98.159|99.103|99.850|99.698|99.848|
|214|97.413|97.083|99.078|99.161|97.716|99.545|99.325|99.543|99.541|
|215|98.294|98.468|99.104|99.699|99.844|99.414|99.691|99.848|99.855|
|217|89.663|91.208|95.217|96.933|98.845|97.802|98.639|99.074|98.601|
|219|98.113|98.850|99.548|99.546|99.327|99.767|99.764|100.0|99.310|
|220|99.290|99.336|100.0|100.0|99.533|100.0|100.0|99.761|99.756|
|221|98.523|99.386|98.422|98.790|99.190|98.962|99.593|99.797|99.393|
|222|97.017|95.588|99.027|97.233|96.414|97.580|98.242|98.282|98.553|
|223|94.163|96.096|98.828|98.807|98.547|98.262|99.268|98.850|99.252|
|228|95.802|97.247|96.955|98.746|98.756|98.730|99.026|98.858|99.754|
|230|98.547|99.777|99.579|100.0|99.116|100.0|100.0|99.774|99.768|
|231|98.675|99.672|100.0|99.658|100.0|99.678|99.691|100.0|100.0|
|232|97.771|99.415|99.085|99.191|98.863|99.438|99.159|99.421|99.726|
|233|91.125|92.192|96.411|97.385|98.327|97.678|99.52|98.705|98.697|
|234|100.0|99.440|99.828|99.642|99.263|99.644|99.621|99.822|99.814|

#### Window Level One Channel Plot

![KNN-Plot-Results](https://i.imgur.com/LwgVC2O.png)

### Incart Database
#### Window Level One Channel
|SIGNAL|SE-10|SE-20|SE-30|SE-50|SE-70|SE-90|SE-110|SE-130|SE-150|
|-|-|-|-|-|-|-|-|-|-|
|I01|79.596|86.453|90.038|94.086|95.109|96.409|98.022|98.545|98.682|
|I02|85.988|89.888|91.902|96.743|98.324|96.636|97.74|97.595|98.249|
|I03|95.464|95.616|98.031|99.014|98.745|99.175|98.758|98.941|99.192|
|I04|88.129|93.631|96.349|98.246|97.951|98.929|98.998|99.023|98.963|
|I05|93.82|94.693|96.073|97.889|98.446|98.529|99.112|97.772|98.851|
|I06|96.939|97.708|98.077|99.396|99.402|99.211|99.408|99.8|99.592|
|I07|96.623|98.879|98.032|99.044|99.425|99.812|99.632|99.814|99.436|
|I08|61.381|76.513|72.257|76.69|77.674|84.615|82.598|87.47|88.564|
|I09|95.11|97.627|97.913|98.783|99.489|99.169|99.203|98.497|99.645|
|I10|93.085|95.961|97.245|97.984|98.543|99.461|99.595|100.0|99.675|
|I11|97.887|98.113|98.387|99.057|98.856|98.768|100.0|99.532|99.762|
|I12|71.555|79.894|83.688|85.589|93.011|93.486|94.128|97.814|97.096|
|I13|98.349|98.93|99.08|98.734|98.995|99.038|99.0|99.753|99.748|
|I14|96.41|97.701|97.222|97.826|98.378|99.718|98.413|98.942|99.174|
|I15|92.789|96.255|97.674|97.561|96.743|98.828|98.868|98.516|98.87|
|I16|92.188|97.134|97.315|97.125|98.02|98.596|98.693|98.007|99.27|
|I17|92.647|95.989|95.666|98.209|99.057|98.538|99.381|98.834|96.793|
|I18|78.311|83.048|87.934|90.244|93.811|94.228|94.581|97.311|97.513|
|I19|93.557|98.627|98.753|99.743|98.585|98.985|99.739|99.225|99.486|
|I20|80.488|84.658|84.888|89.405|90.909|93.284|94.981|94.275|96.923|
|I21|90.023|94.037|95.556|95.551|96.128|97.414|98.643|97.658|98.409|
|I22|83.308|88.853|89.223|90.679|91.354|92.459|95.968|95.71|97.898|
|I23|91.685|96.667|98.268|98.858|99.296|100.0|99.309|99.052|99.093|
|I24|95.481|97.547|97.426|97.905|98.239|99.598|99.187|98.844|99.393|
|I25|96.726|98.466|96.839|98.462|99.408|98.799|98.82|98.551|99.715|
|I26|96.743|97.712|97.106|98.403|98.675|99.057|99.66|99.074|99.671|
|I27|86.162|94.81|95.393|95.926|97.436|98.282|99.182|99.194|98.982|
|I28|96.353|98.571|97.472|98.529|98.442|99.14|99.118|99.415|99.43|
|I29|52.432|59.281|70.334|75.882|82.056|87.5|86.019|88.632|91.075|
|I30|68.776|80.268|81.799|86.789|86.939|87.216|95.052|93.515|94.805|
|I31|47.006|61.261|67.068|75.481|81.703|82.181|88.994|91.111|97.445|
|I32|95.522|97.516|96.997|96.512|97.281|96.904|98.758|97.492|97.771|
|I33|96.657|97.093|97.472|98.263|98.844|98.214|99.16|99.2|99.185|
|I34|93.862|95.724|98.163|99.189|98.241|98.462|99.282|98.974|98.753|
|I35|87.581|93.961|93.693|95.315|96.24|97.937|98.219|99.263|99.833|
|I36|90.659|91.056|93.167|95.238|95.875|96.282|97.568|99.556|99.502|
|I37|96.923|98.773|98.79|99.387|99.387|99.402|99.413|99.143|99.793|
|I38|83.902|92.6|96.127|97.872|96.494|98.493|97.993|98.493|99.242|
|I39|91.343|95.745|97.101|97.953|98.098|98.061|98.3|98.251|99.153|
|I40|94.896|97.426|97.353|98.819|98.521|99.248|99.627|98.672|99.048|
|I41|99.349|99.677|99.023|99.679|99.412|98.684|100.0|99.669|100.0|
|I42|63.107|80.624|84.365|89.386|93.789|95.806|96.474|97.208|97.944|
|I43|85.912|96.674|98.073|97.921|98.301|98.465|97.991|99.523|99.738|
|I44|94.97|98.491|98.416|99.008|98.592|98.592|99.594|99.793|100.0|
|I45|96.373|98.309|98.731|98.082|99.725|99.469|99.457|98.724|98.942|
|I46|86.94|91.603|91.845|94.574|95.146|96.685|96.357|97.719|98.459|
|I47|95.0|96.649|97.222|98.933|98.4|99.0|99.747|99.481|99.486|
|I48|92.744|93.886|97.257|98.337|98.712|99.346|99.788|98.753|99.348|
|I49|97.229|99.558|99.123|100.0|99.275|99.763|99.355|100.0|99.539|
|I50|96.48|97.496|97.424|98.967|99.338|98.969|99.672|98.986|99.654|
|I51|66.603|78.937|85.522|89.418|94.595|96.82|97.574|96.154|98.563|
|I52|96.0|97.983|99.196|98.916|98.792|98.225|99.731|99.734|100.0|
|I53|49.561|74.941|79.545|82.798|89.362|91.087|91.921|97.555|94.647|
|I54|89.89|92.105|96.708|96.429|98.351|98.732|98.901|99.356|98.504|
|I55|91.85|98.551|95.455|97.664|98.658|99.057|98.778|98.84|98.84|
|I56|96.923|97.126|97.753|99.671|98.847|99.72|99.69|98.788|99.133|
|I57|92.555|95.608|97.643|97.138|98.451|98.63|99.649|98.261|98.192|
|I58|96.725|98.131|99.573|99.161|99.129|100.0|99.374|100.0|99.783|
|I59|90.602|92.111|95.392|96.544|95.495|96.912|98.135|98.118|98.533|
|I60|93.996|98.081|98.428|98.478|99.795|98.978|99.231|98.969|98.992|
|I61|98.221|100.0|100.0|99.647|99.671|100.0|100.0|99.644|100.0|
|I62|86.042|91.048|93.833|95.701|97.414|97.773|97.285|97.964|98.413|
|I63|92.911|92.268|96.259|97.625|97.9|96.992|97.216|98.442|97.573|
|I64|88.665|92.545|94.629|97.015|97.594|97.82|99.761|98.765|97.27|
|I65|87.703|91.745|94.297|93.996|97.338|97.043|97.333|98.286|99.594|
|I66|91.142|91.577|95.354|97.55|97.374|97.174|96.849|99.117|98.696|
|I67|77.903|85.161|90.756|91.694|93.31|95.318|95.8|97.217|98.148|
|I68|94.173|96.177|98.336|98.537|99.445|99.454|99.625|99.055|99.81|
|I69|97.959|98.522|99.275|99.074|99.312|99.292|99.546|99.524|98.832|
|I70|94.91|99.023|98.485|100.0|99.713|99.365|100.0|99.096|99.13|
|I71|96.375|99.118|98.584|99.424|99.39|98.266|99.094|99.708|99.69|
|I72|92.76|97.577|97.186|98.391|99.113|99.78|99.557|99.338|99.536|
|I73|96.939|98.54|99.196|99.202|98.99|99.75|99.758|99.492|99.745|
|I74|92.827|96.603|96.687|98.261|98.28|98.077|98.34|98.975|99.793|
|I75|92.905|96.591|97.625|99.536|98.627|99.77|99.01|98.804|99.527|

#### Window Level One Channel Plot

![KNN-Plot-Results](https://i.imgur.com/bN1eEqT.png)

#### Sample Level One Channel
|SIGNAL|SE-10|SE-20|SE-30|SE-50|SE-70|SE-90|SE-110|SE-130|SE-150|
|-|-|-|-|-|-|-|-|-|-|
|I01|79.596|86.453|90.038|94.086|95.109|96.409|98.022|98.545|98.682|
|I02|85.988|89.888|91.902|96.743|98.324|96.636|97.74|97.595|98.249|
|I03|95.464|95.616|98.031|99.014|98.745|99.175|98.758|98.941|99.192|
|I04|88.129|93.631|96.349|98.246|97.951|98.929|98.998|99.023|98.963|
|I05|93.82|94.693|96.073|97.889|98.446|98.529|99.112|97.772|98.851|
|I06|96.939|97.708|98.077|99.396|99.402|99.211|99.408|99.8|99.592|
|I07|96.623|98.879|98.032|99.044|99.425|99.812|99.632|99.814|99.436|
|I08|61.381|76.513|72.257|76.69|77.674|84.615|82.598|87.47|88.564|
|I09|95.11|97.627|97.913|98.783|99.489|99.169|99.203|98.497|99.645|
|I10|93.085|95.961|97.245|97.984|98.543|99.461|99.595|100.0|99.675|
|I11|97.887|98.113|98.387|99.057|98.856|98.768|100.0|99.532|99.762|
|I12|71.555|79.894|83.688|85.589|93.011|93.486|94.128|97.814|97.096|
|I13|98.349|98.93|99.08|98.734|98.995|99.038|99.0|99.753|99.748|
|I14|96.41|97.701|97.222|97.826|98.378|99.718|98.413|98.942|99.174|
|I15|92.789|96.255|97.674|97.561|96.743|98.828|98.868|98.516|98.87|
|I16|92.188|97.134|97.315|97.125|98.02|98.596|98.693|98.007|99.27|
|I17|92.647|95.989|95.666|98.209|99.057|98.538|99.381|98.834|96.793|
|I18|78.311|83.048|87.934|90.244|93.811|94.228|94.581|97.311|97.513|
|I19|93.557|98.627|98.753|99.743|98.585|98.985|99.739|99.225|99.486|
|I20|80.488|84.658|84.888|89.405|90.909|93.284|94.981|94.275|96.923|
|I21|90.023|94.037|95.556|95.551|96.128|97.414|98.643|97.658|98.409|
|I22|83.308|88.853|89.223|90.679|91.354|92.459|95.968|95.71|97.898|
|I23|91.685|96.667|98.268|98.858|99.296|100.0|99.309|99.052|99.093|
|I24|95.481|97.547|97.426|97.905|98.239|99.598|99.187|98.844|99.393|
|I25|96.726|98.466|96.839|98.462|99.408|98.799|98.82|98.551|99.715|
|I26|96.743|97.712|97.106|98.403|98.675|99.057|99.66|99.074|99.671|
|I27|86.162|94.81|95.393|95.926|97.436|98.282|99.182|99.194|98.982|
|I28|96.353|98.571|97.472|98.529|98.442|99.14|99.118|99.415|99.43|
|I29|52.432|59.281|70.334|75.882|82.056|87.5|86.019|88.632|91.075|
|I30|68.776|80.268|81.799|86.789|86.939|87.216|95.052|93.515|94.805|
|I31|47.006|61.261|67.068|75.481|81.703|82.181|88.994|91.111|97.445|
|I32|95.522|97.516|96.997|96.512|97.281|96.904|98.758|97.492|97.771|
|I33|96.657|97.093|97.472|98.263|98.844|98.214|99.16|99.2|99.185|
|I34|93.862|95.724|98.163|99.189|98.241|98.462|99.282|98.974|98.753|
|I35|87.581|93.961|93.693|95.315|96.24|97.937|98.219|99.263|99.833|
|I36|90.659|91.056|93.167|95.238|95.875|96.282|97.568|99.556|99.502|
|I37|96.923|98.773|98.79|99.387|99.387|99.402|99.413|99.143|99.793|
|I38|83.902|92.6|96.127|97.872|96.494|98.493|97.993|98.493|99.242|
|I39|91.343|95.745|97.101|97.953|98.098|98.061|98.3|98.251|99.153|
|I40|94.896|97.426|97.353|98.819|98.521|99.248|99.627|98.672|99.048|
|I41|99.349|99.677|99.023|99.679|99.412|98.684|100.0|99.669|100.0|
|I42|63.107|80.624|84.365|89.386|93.789|95.806|96.474|97.208|97.944|
|I43|85.912|96.674|98.073|97.921|98.301|98.465|97.991|99.523|99.738|
|I44|94.97|98.491|98.416|99.008|98.592|98.592|99.594|99.793|100.0|
|I45|96.373|98.309|98.731|98.082|99.725|99.469|99.457|98.724|98.942|
|I46|86.94|91.603|91.845|94.574|95.146|96.685|96.357|97.719|98.459|
|I47|95.0|96.649|97.222|98.933|98.4|99.0|99.747|99.481|99.486|
|I48|92.744|93.886|97.257|98.337|98.712|99.346|99.788|98.753|99.348|
|I49|97.229|99.558|99.123|100.0|99.275|99.763|99.355|100.0|99.539|
|I50|96.48|97.496|97.424|98.967|99.338|98.969|99.672|98.986|99.654|
|I51|66.603|78.937|85.522|89.418|94.595|96.82|97.574|96.154|98.563|
|I52|96.0|97.983|99.196|98.916|98.792|98.225|99.731|99.734|100.0|
|I53|49.561|74.941|79.545|82.798|89.362|91.087|91.921|97.555|94.647|
|I54|89.89|92.105|96.708|96.429|98.351|98.732|98.901|99.356|98.504|
|I55|91.85|98.551|95.455|97.664|98.658|99.057|98.778|98.84|98.84|
|I56|96.923|97.126|97.753|99.671|98.847|99.72|99.69|98.788|99.133|
|I57|92.555|95.608|97.643|97.138|98.451|98.63|99.649|98.261|98.192|
|I58|96.725|98.131|99.573|99.161|99.129|100.0|99.374|100.0|99.783|
|I59|90.602|92.111|95.392|96.544|95.495|96.912|98.135|98.118|98.533|
|I60|93.996|98.081|98.428|98.478|99.795|98.978|99.231|98.969|98.992|
|I61|98.221|100.0|100.0|99.647|99.671|100.0|100.0|99.644|100.0|
|I62|86.042|91.048|93.833|95.701|97.414|97.773|97.285|97.964|98.413|
|I63|92.911|92.268|96.259|97.625|97.9|96.992|97.216|98.442|97.573|
|I64|88.665|92.545|94.629|97.015|97.594|97.82|99.761|98.765|97.27|
|I65|87.703|91.745|94.297|93.996|97.338|97.043|97.333|98.286|99.594|
|I66|91.142|91.577|95.354|97.55|97.374|97.174|96.849|99.117|98.696|
|I67|77.903|85.161|90.756|91.694|93.31|95.318|95.8|97.217|98.148|
|I68|94.173|96.177|98.336|98.537|99.445|99.454|99.625|99.055|99.81|
|I69|97.959|98.522|99.275|99.074|99.312|99.292|99.546|99.524|98.832|
|I70|94.91|99.023|98.485|100.0|99.713|99.365|100.0|99.096|99.13|
|I71|96.375|99.118|98.584|99.424|99.39|98.266|99.094|99.708|99.69|
|I72|92.76|97.577|97.186|98.391|99.113|99.78|99.557|99.338|99.536|
|I73|96.939|98.54|99.196|99.202|98.99|99.75|99.758|99.492|99.745|
|I74|92.827|96.603|96.687|98.261|98.28|98.077|98.34|98.975|99.793|
|I75|92.905|96.591|97.625|99.536|98.627|99.77|99.01|98.804|99.527|

#### Sample Level One Channel Plot

![KNN-Plot-Results](https://i.imgur.com/nsuduUT.png)


## RPEAK
### Incart
#### window 10
|SIGNAL|DER|SE|DIFF|
|-|-|-|-|
|I01|0.383|77.264|2.593|
|I02|0.02|98.009|0.521|
|I03|0.053|97.419|1.274|
|I04|0.046|97.082|1.511|
|I05|0.08|95.87|1.714|
|I06|0.008|99.601|0.902|
|I07|0.069|96.243|1.05|
|I08|2.483|39.359|1.895|
|I09|0.033|97.407|0.794|
|I10|0.076|95.199|0.954|
|I11|1.196|62.584|4.676|
|I12|2.894|36.444|2.729|
|I13|0.282|85.61|2.53|
|I14|0.13|93.496|1.368|
|I15|0.122|94.151|1.948|
|I16|7.406|21.262|4.016|
|I17|23.64|7.716|1.04|
|I18|1.382|54.927|2.16|
|I19|0.645|66.353|1.663|
|I20|1.393|55.026|2.69|
|I21|2.569|43.046|2.01|
|I22|0.934|62.254|1.498|
|I23|0.027|98.568|1.232|
|I24|0.085|95.446|3.004|
|I25|0.092|95.765|3.061|
|I26|0.0|100.0|3.31|
|I27|1.017|59.369|3.369|
|I28|0.107|93.948|0.601|
|I29|1.265|52.953|2.092|
|I30|1.57|47.401|2.031|
|I31|1.731|43.21|2.045|
|I32|0.086|95.395|1.631|
|I33|0.021|98.98|2.294|
|I34|0.02|98.765|2.058|
|I35|1.446|51.634|1.901|
|I36|0.081|93.95|0.587|
|I37|0.069|95.712|0.788|
|I38|0.929|64.313|1.341|
|I39|0.28|87.702|0.897|
|I40|0.123|93.617|1.331|
|I41|0.024|98.813|1.447|
|I42|1.376|45.556|1.185|
|I43|1.828|39.302|1.562|
|I44|0.39|83.682|1.508|
|I45|0.172|91.842|2.467|
|I46|1.243|57.383|2.368|
|I47|0.219|89.901|1.332|
|I48|0.018|99.109|0.901|
|I49|0.033|98.376|1.533|
|I50|0.005|99.644|1.504|
|I51|1.415|45.769|1.906|
|I52|0.0|100.0|1.079|
|I53|3.993|26.742|2.423|
|I54|0.437|80.077|3.418|
|I55|0.284|87.47|3.975|
|I56|0.0|100.0|3.63|
|I57|0.028|98.507|3.145|
|I58|0.017|98.932|0.689|
|I59|0.242|88.571|0.583|
|I60|0.167|91.459|0.946|
|I61|0.0|100.0|0.517|
|I62|0.113|94.635|0.814|
|I63|0.503|79.198|0.525|
|I64|1.487|56.566|0.304|
|I65|0.198|87.719|1.352|
|I66|0.47|76.507|2.049|
|I67|0.674|69.004|1.545|
|I68|0.243|87.337|2.593|
|I69|0.122|92.593|2.728|
|I70|0.296|86.471|4.194|
|I71|0.131|93.694|3.506|
|I72|0.21|89.731|0.866|
|I73|0.421|82.308|4.308|
|I74|0.266|88.274|0.614|
|I75|0.205|90.692|1.776|

### Window 20
|SIGNAL|DER|SE|DIFF|
|-|-|-|-|
|I01|1.544|51.941|1.929|
|I02|0.02|98.009|0.519|
|I03|3.314|37.634|4.777|
|I04|0.038|97.471|1.505|
|I05|0.093|95.28|1.678|
|I06|0.164|92.415|0.823|
|I07|0.378|83.721|0.959|
|I08|2.168|42.334|1.762|
|I09|0.023|97.893|0.795|
|I10|0.154|91.77|0.895|
|I11|inf|0.0|inf|
|I12|12.681|12.148|2.986|
|I13|11.81|14.146|3.948|
|I14|0.124|93.767|1.185|
|I15|9.899|16.792|2.461|
|I16|8.379|19.269|4.103|
|I17|23.64|7.716|1.2|
|I18|2.922|37.736|1.611|
|I19|0.572|68.235|1.859|
|I20|13.884|11.755|2.913|
|I21|4.275|31.347|3.099|
|I22|1.979|45.915|1.113|
|I23|3.135|38.902|4.675|
|I24|2.188|47.525|3.954|
|I25|203.0|0.977|1.667|
|I26|266.0|0.746|4.5|
|I27|2.054|44.181|3.562|
|I28|0.14|92.507|4.52|
|I29|8.229|16.904|3.349|
|I30|8.853|15.593|2.933|
|I31|2.352|37.037|1.99|
|I32|84.429|2.303|4.0|
|I33|0.047|97.704|2.29|
|I34|0.172|91.852|2.097|
|I35|2.334|41.046|1.669|
|I36|0.062|94.79|0.525|
|I37|0.078|95.322|0.722|
|I38|4.055|31.107|4.252|
|I39|0.28|87.702|4.103|
|I40|0.064|96.277|1.293|
|I41|0.018|99.11|1.446|
|I42|1.296|46.667|0.997|
|I43|2.401|34.186|1.129|
|I44|4.548|30.544|4.726|
|I45|0.094|95.263|2.544|
|I46|1.16|58.893|2.444|
|I47|0.195|90.887|1.154|
|I48|4.027|33.185|4.738|
|I49|7.685|20.65|1.247|
|I50|0.005|99.644|1.477|
|I51|5.07|22.107|0.914|
|I52|0.0|100.0|1.079|
|I53|9.052|14.501|2.182|
|I54|0.369|82.398|3.481|
|I55|2.104|48.675|3.594|
|I56|inf|0.0|inf|
|I57|115.125|1.706|3.375|
|I58|0.026|98.504|0.672|
|I59|0.129|93.265|0.532|
|I60|0.47|80.249|0.647|
|I61|1.36|59.524|0.137|
|I62|0.268|88.197|1.246|
|I63|0.48|79.95|0.429|
|I64|1.352|58.838|0.245|
|I65|0.154|89.474|1.336|
|I66|16.18|10.395|2.66|
|I67|0.545|72.509|1.509|
|I68|0.234|87.709|2.601|
|I69|0.122|92.593|2.728|
|I70|0.336|85.0|4.242|
|I71|3.733|34.835|3.586|
|I72|1.703|53.545|1.269|
|I73|109.0|1.795|2.429|
|I74|0.342|85.398|0.521|
|I75|0.85|70.167|1.252|

### Window 50
|SIGNAL|DER|SE|DIFF|
|-|-|-|-|
|I01|1.57|51.571|1.925|
|I02|0.02|98.009|0.519|
|I03|inf|0.0|inf|
|I04|0.042|97.276|1.512|
|I05|0.086|95.575|1.682|
|I06|0.193|91.218|0.812|
|I07|0.484|80.143|0.893|
|I08|2.123|42.792|1.781|
|I09|0.02|98.055|0.785|
|I10|0.16|91.495|0.871|
|I11|inf|0.0|inf|
|I12|32.931|5.106|2.379|
|I13|131.5|1.463|2.167|
|I14|0.112|94.309|1.178|
|I15|19.18|9.434|1.54|
|I16|8.945|18.272|4.218|
|I17|24.708|7.407|1.042|
|I18|2.978|37.317|1.579|
|I19|0.636|66.588|2.007|
|I20|40.154|4.429|2.731|
|I21|26.742|6.843|2.613|
|I22|1.954|46.197|0.744|
|I23|835.0|0.239|1.0|
|I24|2.589|43.366|3.918|
|I25|66.333|2.932|1.333|
|I26|534.0|0.373|5.0|
|I27|1.676|48.718|3.632|
|I28|17.083|10.375|3.028|
|I29|82.9|2.037|2.7|
|I30|29.308|5.405|1.885|
|I31|2.663|34.568|1.934|
|I32|inf|0.0|inf|
|I33|0.052|97.449|2.275|
|I34|0.27|87.901|2.045|
|I35|3.379|33.072|1.186|
|I36|0.062|94.79|0.523|
|I37|0.073|95.517|0.708|
|I38|195.4|0.954|2.4|
|I39|inf|0.0|inf|
|I40|0.038|97.518|1.276|
|I41|0.012|99.407|1.454|
|I42|1.177|48.413|0.997|
|I43|2.401|34.186|1.129|
|I44|84.909|2.301|2.273|
|I45|0.088|95.526|2.551|
|I46|1.038|61.242|2.367|
|I47|0.16|92.365|1.136|
|I48|inf|0.0|inf|
|I49|8.023|19.954|1.116|
|I50|0.005|99.644|1.477|
|I51|6.303|18.826|0.514|
|I52|0.0|100.0|1.079|
|I53|11.726|11.676|1.758|
|I54|0.278|85.687|3.517|
|I55|2.23|47.229|3.582|
|I56|inf|0.0|inf|
|I57|inf|0.0|inf|
|I58|0.03|98.291|0.663|
|I59|0.053|96.735|0.443|
|I60|0.459|80.605|0.572|
|I61|1.459|57.823|0.141|
|I62|0.348|85.193|1.159|
|I63|0.527|78.446|0.268|
|I64|1.295|59.848|0.245|
|I65|0.133|90.351|1.32|
|I66|37.522|4.782|2.087|
|I67|0.421|76.199|1.506|
|I68|0.187|89.572|2.586|
|I69|0.117|92.824|2.733|
|I70|0.296|86.471|4.259|
|I71|17.559|10.21|4.176|
|I72|3.671|34.963|0.364|
|I73|inf|0.0|inf|
|I74|0.342|85.398|0.521|
|I75|0.86|69.928|1.249|



# Signals' annotations


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


## References 
* 1) [QRS detection using KNN](https://www.researchgate.net/publication/257736741_QRS_detection_using_K-Nearest_Neighbor_algorithm_KNN_and_evaluation_on_standard_ECG_databases) - Indu Saini, Dilbag Singh, Arun Khosla
* 2) [MIT-BIH Arrhythmia Database](https://pdfs.semanticscholar.org/072a/0db716fb6f8332323f076b71554716a7271c.pdf) - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
* 3) [Components of a New Research Resource for Complex Physiologic Signals.](http://circ.ahajournals.org/content/101/23/e215.full) - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet. 
* 4) [WFDB Usages](https://github.com/MIT-LCP/wfdb-python) 
* 5) [QRS complex Detection Algorithm](https://github.com/tru-hy/rpeakdetect/tree/master)



