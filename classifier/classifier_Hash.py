import numpy
import progressbar
# --------------------------------------------------------------------------------------------------------------------
import tools_IO
# --------------------------------------------------------------------------------------------------------------------
class classifier_Hash(object):
    def __init__(self):
        self.name = "Hash"
        self.folder_out = './data/output/'
        return
# ----------------------------------------------------------------------------------------------------------------
    def learn(self, X, Y):

        self.calc_counts(X,Y)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict_one(self, x):
        key = self.feature_to_hash(x)
        if key in self.dct_pos.keys():
            if key not in self.dct_neg:
                result = numpy.array([0,1])
            else:
                result = numpy.array([0, self.dct_pos[key]/(self.dct_pos[key]+self.dct_neg[key])])
        elif key in self.dct_neg.keys():
            result = numpy.array([0, 0])
        else:
            result = numpy.array([0, 0.5])

        return numpy.array([result])
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self, array):
        if len(array.shape)==2 and array.shape[0]>1:
            result = []
            for x in array:
                result.append(self.predict_one(x)[0])
            result = numpy.array(result)
        else:
            result = self.predict_one(array)
        return result
# ----------------------------------------------------------------------------------------------------------------------
    def learn_file(self,filename_in,has_header=False):
        self.calc_counts_file(filename_in, has_header=has_header, delim='\t')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def build_domains(self,filename_in):

        self.domains, success = tools_IO.load_if_exists(self.folder_out + 'domains.dat')

        if success:
            return

        self.domains=[]
        #C = tools_IO.count_columns(filename_in)
        C = 27
        bar = progressbar.ProgressBar(max_value=C)
        for c in range(1,C,1):
            bar.update(c)
            domain={}
            feature = numpy.array(tools_IO.get_columns(filename_in,delim='\t',start=c,end=c+1)).flatten()
            cnt=0
            for val in feature:
                if val not in domain:
                    domain[val]=cnt
                    cnt+=1
            self.domains.append(domain)

        tools_IO.write_cache(self.folder_out + 'domains.dat', self.domains)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def feature_to_hash(self,X):

        len_domains = [len(domain) for domain in self.domains]

        hash_value = 0
        basis = 1
        for n,x in enumerate(X):
            if x in self.domains[n]:
                position = self.domains[n][x]
                hash_value+=basis*position
                basis*=len_domains[n]
            else:
                return -1

        return hash_value
# ---------------------------------------------------------------------------------------------------------------------
    def calc_counts_file(self, filename_in,has_header=False,delim='\t'):

        self.build_domains(filename_in)

        self.dct_pos, success1 = tools_IO.load_if_exists(self.folder_out + 'dct_pos.dat')
        self.dct_neg, success2 = tools_IO.load_if_exists(self.folder_out + 'dct_neg.dat')

        if success1 and success2:
            return

        self.dct_pos,self.dct_neg={},{}

        N = tools_IO.count_lines(filename_in)
        bar = progressbar.ProgressBar(max_value=N)
        with open(filename_in, 'r') as f:
            for i,line in enumerate(f):
                bar.update(i)
                if has_header and i==0:continue
                line = line.strip()
                if len(line) > 0 :
                    X = numpy.array(line.split(delim))
                    Y = (X[0].copy()).astype(int)
                    X = X[1:]

                    hash = self.feature_to_hash(X)

                    if Y>0:
                        if hash in self.dct_pos:
                            self.dct_pos[hash]+=1
                        else:
                            self.dct_pos[hash] = 1
                    else:
                        if hash in self.dct_neg:
                            self.dct_neg[hash]+= 1
                        else:
                            self.dct_neg[hash] = 1

        tools_IO.write_cache(self.folder_out + 'dct_pos.dat', self.dct_pos)
        tools_IO.write_cache(self.folder_out + 'dct_neg.dat', self.dct_neg)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def calc_counts(self, X,Y):

        self.domains=[]

        for c in range(X.shape[1]):
            domain={}
            feature = X[:,c]
            cnt=1
            for val in feature:
                if val not in domain:
                    domain[val]=cnt
                    cnt+=1
            self.domains.append(domain)

        self.dct_pos, self.dct_neg = {}, {}

        for x,y in zip(X,Y):
            hash = self.feature_to_hash(x)

            if y > 0:
                if hash in self.dct_pos:
                    self.dct_pos[hash] += 1
                else:
                    self.dct_pos[hash] = 1
            else:
                if hash in self.dct_neg:
                    self.dct_neg[hash] += 1
                else:
                    self.dct_neg[hash] = 1
        return
# ---------------------------------------------------------------------------------------------------------------------
    def estimate_accuracy(self,filename_in,has_header=False):

        self.calc_counts_file(filename_in,has_header=has_header,delim='\t')

        TP, FP, TN, FN = 0, 0, 0, 0

        for key in self.dct_pos.keys():
            if key in self.dct_neg:
                if self.dct_pos[key] > self.dct_neg[key]:
                    TP += self.dct_pos[key]
                    FP += self.dct_neg[key]
                else:
                    TN += self.dct_neg[key]
                    FN += self.dct_pos[key]
            else:
                TP += self.dct_pos[key]

        for key in self.dct_neg.keys():
            if key not in self.dct_pos:
                TN += self.dct_neg[key]

        accuracy = (TP+TN)/(TP+FP+TN+FN)
        print(accuracy)

        return accuracy
# ---------------------------------------------------------------------------------------------------------------------