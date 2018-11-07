from beatclassification.Preprocessing import Preprocessing
prep = Preprocessing()
classes = ['N', 'V', '/', 'R', 'L', 'A', '!', 'E']
prep.segment(classes)