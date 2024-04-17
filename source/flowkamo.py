import pandas as pd
import numpy as np
from copy import copy,deepcopy
from random import randint
from flowkacol import fcol
from flowkafunc import isinlist, isfloat,unionlist,intersectionlist,nnone,nvl,removelist
from flowkapd import fpd
from sklearn.model_selection import train_test_split


class fmo:
    ALGOS={
        'grbRegr':{'label':'gradient boosting regressor','algo':'regressor'},
        'bagRegr':{'label':'bagging regressor','algo':'regressor'},
        'linRegr':{'label':'linear regression','algo':'regressor'},
        'logRegr':{'label':'logistic regression','algo':'classifier'},
        'polRegr':{'label':'polynomial regression','algo':'regressor','polyfit':True},
        'knn':{'label':'k-nearest neighbors','algo':'classifier'},
        'ann':{'label':'artificial neural network','algo':'classifier'},
        'cnn':{'label':'convolutional neural network','algo':'classifier'},
        'rnn':{'label':' recurrent neural network','algo':'classifier'},
        'kmns':{'label':'k-means','algo':'cluster','fitpredict':True},
        'forRegr':{'label':'random forest regressor','algo':'regressor'},
        'treeRegr':{'label':'decision tree regressor','algo':'regressor'},
        'bagClass':{'label':'bagging classifier','algo':'classifier'},
        'forClass':{'label':'random forest classifier','algo':'classifier'},
        'treeClass':{'label':'decision tree classifier','algo':'classifier'},
        'grbClass':{'label':'gradient boosting classifier','algo':'classifier'},
        'svm':{'label':'support vector machine','algo':'classifier'},
        'mns':{'label':'mean shift','algo':'autocluster'},
        'dbs':{'label':'dbscan','algo':'autocluster'},
        'afp':{'label':'afinProp','algo':'autocluster'},
        'spcl':{'label':'spClust','algo':'cluster','fitpredict':True}
        }

    def alg_fitpredict(self,code):
        try:
            return self.ALGOS[code].get('fitpredict',False)
        except:
            return False

    def alg_polyfit(self,code):
        try:
            return self.ALGOS[code].get('polyfit',False)
        except:
            return False

    def alg_label(self,code):
        try:
            return self.ALGOS[code]['label']
        except:
            return False

    def alg_algo(self,code):
        try:
            return self.ALGOS[code]['algo']
        except:
            return False

    def alg_codelist(self,algo):
        return [key for  key,value in self.ALGOS.items() if value.get('algo')==algo]

    def alg_print(self,code):
         print('{} : {} : {}'.format(code,self.ALGOS[code]['label'],self.ALGOS[code]['algo']))

    def __init__(self,ds=None):
        self._ds=ds #the Flowka dataframe extension
        self._model=None
        self._knn_max=10000 #maximum values for knn algorithm
        self._kmns_max=10 #maximum values for knn algorithm
        self.Xy()
        self._test_size=0.3
        self.algo=None
        self._scaler=None

    def _set_test_size(self,test_size=0.3):
        """set the ratio of the test  and the train sets
        """
        self._test_size=test_size


    def _get_test_size(self):
        """get the ratio of the test  and the train sets
        """
        return self._test_size

    test_size=property(_get_test_size,_set_test_size)

    def Xy(self):
        """create a X and a y dataframe
        """
        if len(self.ds.targets) > 0:
            self.y=self.ds.df[self.ds.targets]
            self.X=self.ds.df.drop(self.ds.targets,axis=1)
        else:
            self.X=self.ds.df
            self.y=None

    def __get_ds(self):
        return self._ds

    def __set_ds(self,ds=None):
        if ds is None:
            raise ValueError("dataset should exist")
        else:
            self._ds = ds
            self.Xy()


    ds = property(__get_ds,__set_ds)

    def save_model(self,model_file=None):
        """export model to a file for later use
           model_file : set a specific model file; by default it is set to local 'flowka_model_*name.csv'
        """
        import pickle
        if model_file is None:
            model_file = './work/flowka_model_'+self.ds.name+'.dat'
        else:
            model_file=str(model_file).strip()
        try:
            columnorder=self.X.columns.tolist()#to keep the column order identical thru datasets
            pickle.dump({'model':self._model,'scaler':self._scaler,'algo':self._algo, 'columnsorder':columnorder}, open(model_file, 'wb'))
            #add to dict : 'normalizer':self._normalizer
            print("model {0} saved".format(model_file))
        except ValueError:
            print("unable to save model {0}".format(model_file))

    def load_model(self,model_file=None):
        """export model to a file for later use
           model_file : set a specific model file; by default it is set to local 'flowka_model_*name.csv'
        """
        import pickle
        if model_file is None:
            model_file = './work/flowka_model_'+self.ds.name+'.dat'
        else:
            model_file=str(model_file).strip()
        try:
            load_dict = pickle.load(open(model_file, 'rb'))
            print("model {0} loaded".format(model_file))
        except:
            print("model {0} not found".format(model_file))

        try:
            self._model=load_dict['model']
            self._scaler=load_dict['scaler']
            self.algo=load_dict['algo']
            self.X=self.X[load_dict['columnsorder']]
            #self._normalizer=load_dict['normalizer']
        except:
                print("format {0} is invalid".format(model_file))
        if self._scaler is not None:
            self.rescale()


    def scale(self,scaler='std'):
        """standardize an normalize data
           standardize : center data around zero -> (Xi-mean(Xi))/StdDev(Xi)
           normalize : fit data around -1 and 1 -> (Xi/(max(Xi)-min(Xi)))
           scaler : 'std', 'robust', uniform'44
        """
        if scaler == 'std':
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
        elif scaler == 'robust':
            from sklearn.preprocessing import RobustScaler
            self._scaler = robustScaler(quantile_range=(10, 90))
        else:
            from sklearn.preprocessing import QuantileTransformer
            self._scaler=QuantileTransformer(output_distribution='uniform')
        #from sklearn.preprocessing import Normalizer
        #self._normalizer=Normalizer()
        #self.X=self._normalizer.fit_transform(self.X)
        columns=self.X.columns
        self.X=self._scaler.fit_transform(self.X)
        self.X=pd.DataFrame(self.X, columns=columns)

    def rescale(self):
        """re-execute the scaling
        """
        #self.X=self._normalizer.fit_transform(self.X)

        if self._scaler is None:
            return self.X
        columns=self.X.columns
        self.X=self._scaler.transform(self.X)
        self.X=pd.DataFrame(self.X, columns=columns)


    def unscale(self):
        """inverse scale transformation
        """
        columns=self.X.columns
        self.X=pd.DataFrame(self._scaler.inverse_transform(self.X), columns=columns)

    def linRegr(self):
        """Linear Regression
        """
        self.alg_print('linRegr')
        from sklearn.linear_model import LinearRegression
        self._model = LinearRegression()
        self.split(self.X, self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def logRegr(self):
        """Logistic Regression
        """
        self.alg_print('logRegr')
        from sklearn.linear_model import LogisticRegression
        self._model=LogisticRegression()
        self.split(self.X, self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def polyfit(self,Xdata):
        """fit polynomial regression
        """
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg=PolynomialFeatures(degree=4)
        Xpoly=poly_reg.fit_transform(Xdata)
        return Xpoly

    def polRegr(self):
        """Polynomial regression
        """
        #X-datas (X-test, X-train... should be fitted with polyfit() before fit() and predict())
        self.alg_print('polRegr')
        from sklearn.linear_model import LinearRegression
        self._model=LinearRegression()
        self.split(self.X, self.y)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)


    def knn(self,k=None):
        """k-Nearest Neighbors
        """
        self.alg_print('knn')
        self.split(self.X, self.y)
        from sklearn.neighbors import KNeighborsClassifier
        if k is None:
            error_rate=[]
            preverr=[0,1]
            for k in range (1,30): #test over 30 values of k
                self._model=KNeighborsClassifier(n_neighbors=k)
                self.fit(self.X_train,self.y_train)
                self.predict(self.X_test)
                err=(np.mean(self._predictions != self.y_test))
                error_rate.append
                if err<preverr[1]: #if the error is lower than the lowest error stored
                    preverr=[k,err]#we store the new k and its error value
            print("k = {0} - error rate = {1}".format(str(preverr[0]),str(preverr[1])))
            k=preverr[0]
        self._model=KNeighborsClassifier(n_neighbors=k)#we choose the k value with the lowest error rate
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)#redo the prediction with chosen k

    def svm(self):
        """Support Vector Machine
        """
        self.alg_print('svm')
        self.split(self.X, self.y)
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        param_grid={'C':[0.1,0.5,1,5,10,50,100],'gamma':[1,0.5,0.1,0.05,0.01,0.005,0.001]}
        self._model = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)#seearch the best parameters
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test) #fit and predict

    def cnn(self):
        self.alg_print('cnn')
        self.split(self.X, self.y)
        self._prediction= pd.Series(np.zeros(self.y_test.shape[0]))

    def rnn(self):
        self.alg_print('rnn')
        self.split(self.X, self.y)
        self._prediction= pd.Series(np.zeros(self.y_test.shape[0]))

    def ann(self):
        self.alg_print('ann')
        self.split(self.X, self.y)
        self._prediction= pd.Series(np.zeros(self.y_test.shape[0]))

    def kmns(self,clusters=None):
        """K-Means
        """
        self.alg_print('kmns')
        self.clusters=nvl(nvl(clusters,self.ds.clusters),2)#if no clsuters specified
        from sklearn.cluster import KMeans
        self._model = KMeans(n_clusters = self.clusters, init = 'k-means++', random_state = 0)
        self.fit(self.X)
        self.predict(self.X)
        #add predictions as a new column to the dataset
        self.ds.add=pd.DataFrame(self._predictions,columns=['kMeans'])

    def spcl(self,clusters=None):
        """Spectral Clustering
        """
        self.alg_print('spcl')
        from sklearn.cluster import SpectralClustering
        self.clusters=nvl(nvl(clusters,self.ds.clusters),2)#if no clsuters specified
        self._model = SpectralClustering(n_clusters = self.clusters, eigen_solver='arpack',affinity="nearest_neighbors")
        self.fit(self.X)
        self.predict(self.X)
        #add predictions as a new column to the dataset
        self.ds.add=pd.DataFrame(self._predictions,columns=['SpectralClustering'])


    def afp(self):
        """Affinity Propagation
        """
        self.alg_print('afp')
        from sklearn.cluster import AffinityPropagation
        self._model = AffinityPropagation()
        self.fit(self.X)
        #add predictions as a new column to the dataset
        self.predict()
        self.ds.add=pd.DataFrame(self._predictions,columns=['AffinityPropagation'])

    def mns (self):
        """Mean Shift
        """
        self.alg_print('mns')
        from sklearn.cluster import MeanShift,estimate_bandwidth
        self._model = MeanShift()
        self.fit(self.X)
        #add predictions as a new column to the dataset
        self.predict()
        self.ds.add=pd.DataFrame(self._predictions,columns=['meanShift'])

    def dbs (self):
        """DBScan
        """
        self.alg_print('dbs')
        from sklearn.cluster import DBSCAN
        lmlist=self.X.columns
        self._model = DBSCAN()
        self.fit(self.X)
        #add predictions as a new column to the dataset
        self.predict()
        self.ds.add=pd.DataFrame(self._predictions,columns=['dbScan'])


    def forClass(self):
        """Random Forest Classifier'
        """
        self.alg_print('forClass')
        self.split(self.X, self.y)
        from sklearn.ensemble import RandomForestClassifier
        self._model=RandomForestClassifier(n_estimators=600)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)


    def treeClass(self):
        """Decision Tree Classifier
        """
        self.alg_print('treeClass')
        self.split(self.X, self.y)
        from sklearn.tree import DecisionTreeClassifier
        self._model=DecisionTreeClassifier()
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def forRegr(self):
        """Random Forest Regressor'
        """
        self.alg_print('forRegr')
        self.split(self.X, self.y)
        from sklearn.ensemble import RandomForestRegressor
        self._model=RandomForestRegressor(n_estimators=600)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def treeRegr(self):
        """Decision Tree Regressor
        """
        self.alg_print('treeRegr')
        self.split(self.X, self.y)
        from sklearn.tree import DecisionTreeRegressor
        self._model=DecisionTreeRegressor( min_samples_leaf=10,
            min_samples_split=10)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def bagRegr(self):
        """Bagging Regressor
        """
        self.alg_print('bagRegr')
        self.split(self.X, self.y)
        from sklearn.ensemble import BaggingRegressor
        self._model = BaggingRegressor(
            n_estimators=600,max_features=0.9)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def bagClass(self):
        """Bagging Classifier
        """
        self.alg_print('bagClass')
        self.split(self.X, self.y)
        from sklearn.ensemble import BaggingClassifier
        self._model = BaggingClassifier(n_estimators=600, max_features=0.9)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def grbRegr(self):
        """Gradient Boosting Regressor
        """
        self.alg_print('grbRegr')
        self.split(self.X, self.y)
        from sklearn.ensemble import GradientBoostingRegressor
        self._model = GradientBoostingRegressor(loss='lad',
            n_estimators=600, max_depth=None,
            learning_rate=.1, min_samples_leaf=10,
            min_samples_split=10)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def grbClass(self):
        """Gradient Boosting Classifier
        """
        self.alg_print('grbClass')
        self.split(self.X, self.y)
        from sklearn.ensemble import GradientBoostingClassifier
        self._model = GradientBoostingClassifier(loss='deviance',
            n_estimators=600, max_depth=5,
            learning_rate=.1, min_samples_leaf=10,
            min_samples_split=10)
        self.fit(self.X_train,self.y_train)
        self.predict(self.X_test)

    def predict(self,inputs=None):
        """predict results from inputs
        """
        inputs=[inputs,self.X][inputs is None]
        print('prediction over {0} rows'.format(inputs.shape[0]))
        if self.alg_polyfit(self._algo):
            inputs=self.polyfit(inputs)
        if self.alg_fitpredict(self._algo):
            self._predictions=self._model.fit_predict(inputs)
        elif self.alg_algo(self._algo)=='autocluster':
            self._predictions=self._model.labels_
            self.clusters= np.shape(np.unique(self._predictions))[0]
        else:
            self._predictions=self._model.predict(inputs)

        return  self._predictions

    def fit(self,X,y=None):
        """fit model over input and output training data
        """
        if self.alg_polyfit(self._algo):
            X=self.polyfit(X)

        if y is None:
            self._model.fit(X)
        else:
            self._model.fit(X,y)

    def split(self,X,y):
        """split dataset in training and testing data
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=self.test_size, random_state=101)

    def pred_append(self,name='predictions'):
        """return predictions into a dataframe
        """
        self.ds.df=self.ds.df.reset_index()
        self.ds.df['predictions']=pd.DataFrame(self._predictions,columns=[name])

    def __get_algo(self):
        return self._algo

    def __set_algo(self,algo):
        """set the algorithm and train the model
        """
        self._algo = algo

    algo = property(__get_algo,__set_algo)

    def train(self,algo=None):
            """set the algorithm and train the model
            """
            print("training")
            algo = algo or self._algo
            if not self.alg_label(algo):
                raise ValueError("algo does not exist")
            eval('self.'+algo+'()')

    def __get_lrcoef(self):
        """give the coefficients of a linear regression
        """
        self._lrcoef=pd.DataFrame(self._model.coef_,self.ds.df.drop(self.ds.targets,axis=1).columns)
        self._lrcoef.columns = ['Coeffecient']
        return self._lrcoef

    lrcoef =property(__get_lrcoef)

    def regr_error(self,y,pred):
        dif=pd.DataFrame(np.array([y,pred]).T,columns=['real','pred'])
        dif['dif']=abs(dif['real']-dif['pred'])
        sum=dif.sum(axis=0)
        return sum['dif']/sum['real']

    def metrics(self):
        """evaluate the model performance by calculating
           the residual sum of squares and the explained variance score (R^2).
           Mean Absolute Error - Meaself.Xy(test=False)n Squared Error - Root Mean Squared Error
           Confusion Matrix :
                          Predicted True / Predicted False
           observed True      TN               FP
           observed False     FN               TN

           PRECISION : (True) TP/(FP+TP) *** (False) TN/(FN+TN)
           RECALL : (True) TP/(TP+FN) (sensitivity) *** (False) TN/(TN+FP) (specificity)
           F1-SCORE : 2xPRECISIONxRECALL/(PRECISION+RECALL)
           ACCURACY : (TP+TN)/(TP+TN+FP+FN)
           ERROR RATE = (FP+FN)/(TP+TN+FP+FN)
        """
        from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, silhouette_score
        self._metrics={'MAE':None,'MSE':None,'RMSE':None,'CM':None,'AR':None,'ER':None,'SC':None,'Error':None}
        if self._algo in self.alg_codelist('classifier'):
            cm=confusion_matrix(self.y_test,self._predictions,labels=list(set(self.y_test)))
            ar=cm.trace()/cm.sum()# accuracy rate -> sum(diag)/sum()
            er=(cm.sum()-cm.trace())/cm.sum()#error rate -> (sum()-sum(diag))/sum()
            self._metrics.update({'CM':cm,'AR':ar,'ER':er,'Error':er})
        elif self._algo in self.alg_codelist('regressor'):
            mae=mean_absolute_error(self.y_test, self._predictions)
            mse=mean_squared_error(self.y_test, self._predictions)
            rmse=np.sqrt(mse)
            rer=self.regr_error(self.y_test, self._predictions)
            self._metrics.update({'MAE':mae,'MSE':mse,'RMSE':rmse,'Error':rmse, 'RER':rer})
        elif self._algo in self.alg_codelist('cluster')+self.alg_codelist('autocluster'):
            if np.shape(np.unique(self._predictions))[0]<2:
                sc=99999
            else:
                sc=silhouette_score(self.X, self._predictions)
            self._metrics.update({'Error':sc,'SC':sc})
        else:
            none=None
        return self._metrics

    def reports(self):
        """print the Metrics
        """
        self.metrics()
        if self._algo in self.alg_codelist('classifier'):
            from sklearn.metrics import classification_report
            print(pd.DataFrame(data=self._metrics['CM'],columns=list(set(self.y_test)),index=list(set(self.y_test))))
            print("\n")
            print(classification_report(self.y_test,self._predictions))
            print("\n")
            print ('accuracy rate \t: {0}\nerror rate \t: {1}'.format(self._metrics['AR'],self._metrics['ER']))
            print("\n")
        elif self._algo in self.alg_codelist('regressor'):
            print ('Mean Absolut Error (MAE) : {0}\nMean Squared Error (MSE) : {1}\nRoot Mean Squared Error (RMSE) : {2}'.format(self._metrics['MAE'],self._metrics['MSE'],self._metrics['RMSE']) )
            print ('Regression Error Rate : (RER) {0}'.format(self._metrics['RER']) )
            if self._algo =='linRegr':
                print(self.lrcoef) #coef de regression
                #residual plot
        elif  self._algo  in self.alg_codelist('cluster')+self.alg_codelist('autocluster'):
                print ('Silhouette Score (SC) : {0}\n'.format(self._metrics['SC']))

    def detect_algo(self):
        """ detect the ml algo based on targets category and values
        """
        algolist=[]
        if len(self.ds.targets)==0 and self.ds.clusters is not None:
            algolist.extend(self.alg_codelist('cluster')) #clustering cluster is known
        elif len(self.ds.targets)==0 and self.ds.clusters is None:
            algolist.extend(self.alg_codelist('autocluster')) #clustering cluster is not known
        elif (str(self.ds.details.loc['dtype',self.ds.targets]).find('int')>-1 and self.ds.details.loc['distratio',self.ds.targets]<0.1) or (str(self._ds.details.loc['dtype',self._ds.targets])=='category'):
            algolist.extend(removelist(self.alg_codelist('classifier'),['logRegr','knn'])) #classification
            if self.ds.details.loc['max',self.ds.targets]==1 and self.ds.details.loc['distinct',self.ds.targets]==2:
                algolist.extend(['logRegr'])#logistic regression
            if (self.ds.details.loc['count',self.ds.targets]<self._knn_max) and (self.ds.details.loc['distinct',self.ds.targets]<10):
                algolist.extend(['knn'])#knn classification
                pass
        elif isinlist(self.ds.targets,self.ds.dtypes['continuous']):
            if self.ds.df.shape[1]<20:
                algolist.extend(self.alg_codelist('regressor'))#regression with limited columns
            else:
                algolist.extend(removelist(self.alg_codelist('regressor'),['linRegr','polRegr']))#regression with limited columns

        print("algos : {0}".format(algolist))
        return algolist

    def best_predict(self,algolist):
        """ test all available algorithm for a data problem and keep the best (less Mean Square Error)
        """
        if self.ds.df.shape[0]>2000:
            X_temp=self.X
            y_temp=self.y
            self.X=self.X.sample(n=2000,random_state=105)
            if self.y is not None:
                self.y=self.y.sample(n=2000,random_state=105)

        bestpredict=[]
        if algolist is None:
            print("No algorithms to test")
            return -1
        elif len(algolist)>1:
            for testalgo in algolist:
                print(testalgo)
                self._algo=testalgo
                self.train(testalgo)

                RER=self.metrics().get('RER',None)
                if RER:
                    print("Mean Regressor Error Rate :{0}".format(RER))

                Error=self.metrics()['Error']
                print("Error:{0}".format(Error))
                if len(bestpredict)==0 or Error<bestpredict[1]:
                    bestpredict=[testalgo,Error]
        else :
            bestpredict.extend(algolist)
        print("choose : {0}\n".format(bestpredict[0]))
        self.algo=bestpredict[0]# set the best algo

        if self.ds.df.shape[0]>2000:
            self.X=X_temp
            self.y=y_temp
