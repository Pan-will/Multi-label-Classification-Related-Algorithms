from numpy import *
from .transforms import *
import copy
from .LP import LP

class ML(LP) :
    '''
        Meta-Label Classifier
        --------------------------
        Essentially 'RAkELd'; need to add pruning option (inherit PS instead of LP), 
        to make it a generic meta-label classifier.
    '''

    k = 3

    def fit(self, X, Y):
        Yy,self.reverse = transform_BR2ML(Y,self.k)
        N, self.L = Y.shape
        N_, L_ = Yy.shape
        from BR import BR
        self.h = BR(L_,copy.deepcopy(self.h))
        self.h.fit(X,Yy)
        return self

    def predict(self, X):
        '''
            return predictions for X
        '''
        Yy = self.h.predict(X)
        N,D = X.shape
        Y = transform_ML2BR(Yy,self.reverse,self.L,self.k)
        return Y

def demo():
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    from sklearn import linear_model
    h = linear_model.LogisticRegression()
    h = linear_model.SGDClassifier(n_iter=100)
    ml = ML(L, h)
    ml.fit(X, Y)

    # Eval
    print(ml.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

