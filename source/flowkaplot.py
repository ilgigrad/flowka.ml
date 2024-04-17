import matplotlib.pyplot as plt
import seaborn as sns
from flowkacol import fcol
import matplotlib.pyplot as plt
import itertools as it
sns.set_style('whitegrid')


class fplot:
    "extend dataframe for Flowka"

    def __init__(self,ds=None,name=''):
        self.ds=ds
        self.fcol=fcol()
        self._maxshowsamples=2000 #number ox maximum samples for graphic display

    def _get_maxshowsamples(self):
        """ return the number ox maximum samples for graphic display
        """
        return copy(self._maxshowsamples)

    def _set_maxshowsamples(self,maxshowsamples):
        """ set the number ox maximum samples for graphic display
        """
        self.log("_set_maxshowsamples({0})".format(maxshowsamples))
        if not isfloat(maxshowsamples):
            raise ValueError("float expected for maxim show samples value")
        else:
            self._maxshowsamples = maxshowsamples



    maxshowsamples = property(_get_maxshowsamples,_set_maxshowsamples)



    def pairplot(self,columns=None):
        """display a pairplot graph of all numerical/contiuous columns
           pairplot shows correlations between variables wether they are linear or not
           columns= list of columns to display to bypass default selection
        """
        # prevent large pairplot of boolean columns/ booleans are only used for 'hue' parameter as a targets
        #tempdf=self.df.sample(n=self.maxshowsamples)
        tempdf=self.ds.df.sample(n=self._maxshowsamples,replace=True)
        targets=[self.targets,None] [len(self.targets)==0]
        if columns is None:
            details=self.ds.details.transpose()
            columns=intersectionlist(self.ds.dtypes['continuous'],list(details[details['distinct']>2].index))
            details=None
        else:
            columns=intersectionlist(columns,list(tempdf.columns))
        if targets is not None and not isinlist([targets],columns):
            columns.append(targets)
        g = sns.PairGrid(tempdf[columns], hue=targets,dropna=True,palette=self.fcol.palette)
        g = g.map_diag(plt.hist,bins=20)
        g = g.map_offdiag(plt.scatter)
        if targets is not None and tempdf[targets].nunique()<6: #do no build a colorization/hue of targets column if more than 5 distinct values
            g = g.map_lower(sns.kdeplot,cmap=self.fcol.cmap)
            g = g.add_legend()
        plt.show()


    def lmplot(self,columns=None):
        """display a lmplot graph of all numerical columns
           by default it displays only columns with valratio > 0.2
           columns: force a list of columns to display; those columns have to be continuous
        """
        #tempdf=self.df.sample(n=self.maxshowsamples)
        tempdf=self.ds.df.sample(n=self._maxshowsamples,replace=True)
        targets=[self.ds.targets,None] [len(self.ds.targets)==0]
        if columns is None:
            details=self.ds.details.transpose()
            columns=intersectionlist(self.ds.dtypes['continuous'],list(details[details['distratio']>0.05].index))
            #columns=self.dtypes['continuous']
            details=None
        else:
            columns=intersectionlist(columns,self.ds.dtypes['continuous'])
        #do no build a colorization/hue of targets column if more than 5 distinct values
        if  targets is not None and tempdf[targets].nunique()<6:
            huelegend= True
            try:
                columns.remove(targets)
            except:
                False
        else:
            huelegend= False
        for lm in it.combinations(columns,2):
                ax=sns.lmplot(x=lm[0],y=lm[1],data=tempdf,hue=targets,size=6,aspect=1,fit_reg=False,palette=self.fcol.palette, legend=huelegend)
                ax=sns.kdeplot(tempdf[lm[0]],tempdf[lm[1]],zorder=0,n_levels=10,shade=True,cmap=self.fcol.cmap)



    def heatmap(self,columns=None):
        """display a heatmap all columns
           columns : for a specific list of columns to set in correlation map
           heatmap point out only linear correlations / it does not show non linear correlation wether they are monotonic or not
        """
        #tempdf=self.df.sample(n=self.maxshowsamples)
        #tempdf=self.df.sample(n=self._maxshowsamples)
        tempdf=self.ds.df
        if columns is None:
            columns = tempdf.columns
        plt.figure(figsize=(10,10))
        sns.heatmap(tempdf[columns].corr(method='pearson'),cmap=self.fcol.cmap)
        plt.title(self._name)

    def histmap(self,columns=None):
        """display a distribution map of each columns
        """
        targets=[self.ds.targets,None] [len(self.ds.targets)==0]
        if columns is None:
            columns = list(self.ds.df.columns)

        if targets is not None and (self.ds.df[targets].nunique()==2):
            try:
                columns.remove(targets)
            except:
                False
            for element in columns:
                plt.figure(figsize=(10,10))
                color=self.fcol.multicolor(2)
                self.ds.df[self.ds.df[targets]==self.ds.details.loc['min',self.ds.targets]][element].hist(alpha=0.7,
                    color=color[0],bins=30,
                    label=targets+'='+str(self.ds.details.loc['min',targets]),
                    edgecolor=self.fcol.c())
                self.ds.df[self.ds.df[targets]==self.ds.details.loc['max',targets]][element].hist(alpha=0.7,
                    color=color[1],bins=30,
                    label=targets+'='+str(self.ds.details.loc['max',targets]),
                    edgecolor=self.fcol.c())
                plt.legend()
                plt.xlabel(element)
        else:
            for element in columns:
                g = sns.FacetGrid(self.ds.df,size=4,aspect=2)
                g.map(plt.hist,element,bins=30,edgecolor=self.fcol.c(),color=self.fcol.color)
        plt.title(self.ds._name)



    def predictplot(self,model):
        """Create a scatter plot of the real test values versus the predicted values.
           if targets is not binary nor classification : scatterplot
           else confusion matrix
        """
        if self.ds.details.loc['distinct',self.ds.targets]>2 and str(self.ds.details.loc['dtype',self.ds.targets])!='category':
            sns.color_palette=self.fcol.palette
            sns.edgecolor=self.fcol.c('flowkadark')
            sns.regplot(x=model.y_test, y=model._predictions, color=self.fcol.color)


    def plotreports(self,model):
        """print the Metrics
        """
        if model._algo in ['logRegr','knn','forClass','treeClass','svm','grbClass','bagClass']:
            sns.heatmap(model._metrics['CM'],cmap=self.fcol.cmap,annot=True,fmt='.0f')
            plt.xlabel('Prediction')
            plt.ylabel('Observation')
        elif model._algo in ['linRegr','forRegr','treeRegr','polRegr','grbRegr','bagRegr']:
            self.predictplot(model)
            if model._algo in ['linRegr']:
                sns.distplot(model.y_test-model._predictions,bins=50,kde=False,color=self.fcol.color)
        plt.show()

    def clusterplot(self,model):
        """plot cluster prediction
        """
        import pandas as pd
        model.alg_print(model._algo)

        palette=self.fcol.palette
        df=pd.concat([model.X,pd.DataFrame(data=model._predictions,columns=[model._algo])],axis=1)
        for lm in it.combinations(model.X.columns,2):
            ax=sns.lmplot(x=lm[0],y=lm[1],data=df,hue=[model._algo,None][model.clusters==1],size=3,aspect=1,fit_reg=False,palette=palette,legend=True)
        plt.title('{0} clusters for {1}'.format(str(model.clusters),model._algo), size=18)
        plt.show()
