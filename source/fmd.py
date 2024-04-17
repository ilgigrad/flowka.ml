import pandas as pd
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from core.utils import isinlist, isfloat,unionlist,intersectionlist
from uuid import uuid4
from copy import copy,deepcopy

class fmd:
    "metadata for Flowka"
    def __init__(self,df,name):
        self.name=name
        self.nb_column=len(list(df.columns))
        self.nb_float=len(list(df.select_dtypes(include=[float]).columns))
        self.nb_int=len(list(df.select_dtypes(include=[int]).columns))
        self.nb_bool=len(list(df.select_dtypes(include=[bool]).columns))
        self.nb_object=len(list(df.select_dtypes(include=[object]).columns))
        self.nb_category=len(list(df.select_dtypes(include=['category','object']).columns))
        self.nb_continuous=len(list(df.select_dtypes(exclude=['category','object']).columns))
        self.nb_time=None
        self.column_list=list(df.columns)
        self.targets_column=''
        self.nb_clusters=None
        self.datatype_targets=None
        self.distinct_targets=None
        self.distratio_targets=None
        self.max_targets=None
        self.min_targets=None
        self.most_frequent_targets=None
        self.less_frequent_targets=None
        self.median_frequent_targets=None
        self.mean_targets=None
        self.stdev_targets=None
        self.q10_targets = None
        self.q25_targets = None
        self.q50_targets = None
        self.q75_targets = None
        self.q90_targets = None
        self.drop_list=None
        self.dummies_list=None
        self.ordinals_list=None
        self.frequencies_list=None
        self.category_list=None
        self.model=None
        self._algo=None
        self.df=df
        self.metadata=self.metadata()


    def load(self,mdfile=None):
        """ load metadata from file
            by default, load from local 'flowka_metadata.dat'
            mdfile: set a specific file...
        """
        import os.path
        mdfile=(str(mdfile).split('.')[0].strip()+'.dat','./work/flowka_metadata_info.dat')[mdfile is None]
        print ("import metadata from {0}".format(mdfile))
        if os.path.isfile(mdfile):
            loader=pd.read_csv(mdfile)
            lastrow=loader[loader['name']==self.name]['time'].max()
            infodf=loader[loader['time']==lastrow]
            if len(infodf)>0:
                infodict=infodf.to_dict('records')[0]
                for key, values in infodict.items():
                    if key not in ['df','metadata']:
                        self.__dict__[key] = values
            else:
                print("no records for '{0}' in {1} ".format(self.name,mdfile))
        else:
            print("file {0} does not exists".format(mdfile))


    def save(self,mdfile=None):
        """export columns parameters to a metadata file for further machine learning on columns classification
           mdfile : set a specific metadata file; by default it is set to local 'flowka_metadata.csv'
        """
        import os.path
        from time import time
        self.id=uuid4()
        self.time=time()
        info=self.__dict__.copy()
        targetsdict={}
        for key,value in info.items():
            if key in ['df','metadata']:
                False
            elif type(value)==list:
                targetsdict[key]=[value]
            else:
                targetsdict[key]=value
        targetsdf=pd.DataFrame(targetsdict,index=[0])
        mdfile=(str(mdfile).split('.')[0].strip(),'./work/flowka_metadata')[mdfile is None]
        header= False if os.path.isfile(mdfile+'_info.dat') else True
        targetsdf.to_csv(open(mdfile+'_info.dat', 'a'), header=header,index=False)
        header= False if os.path.isfile(mdfile+'_detail.dat') else True
        self.metadata.to_csv(open(mdfile+'_detail.dat', 'a'), header=header,index=False)
        print ("metadata exported to {0}".format(mdfile))



    def metadata(self):
        info=pd.DataFrame({'dtype': self.df.dtypes,
                           'min': self.df.min(),
                           'max': self.df.max(),
                           'mean': self.df.mean(),
                           'stddev': np.sqrt(self.df.var()),
                           '10%': self.df.quantile(0.10,interpolation='lower'),
                           '25%': self.df.quantile(0.25,interpolation='lower'),
                           '50%': self.df.quantile(0.5,interpolation='midpoint'),
                           '75%': self.df.quantile(0.75,interpolation='higher'),
                           '90%': self.df.quantile(0.90,interpolation='higher'),
                           'nan': self.df.isnull().sum(),
                           'count': self.df.count(),
                           'valid': self.df.count()+self.df.isnull().sum(),
                           'distinct': self.df.nunique(),
                           'distratio': self.df.nunique()/(self.df.count()+self.df.isnull().sum()),
                           'category': self.df.dtypes=='object',
                           'continuous':self.df.dtypes!='object',
                           'object':self.df.dtypes=='object',
                           'bool': self.df.dtypes=='bool',
                           'int': self.df.dtypes=='int',
                           'float': self.df.dtypes=='float',
                           'time': self.df.dtypes==False,
                           'targets': self.df.dtypes==False,
                           'drop': self.df.dtypes==False,
                           'rename': self.df.dtypes==False,
                           'transform': self.df.dtypes==False,
                           'dummies': self.df.dtypes==False,
                           'ordinals':self.df.dtypes==False,
                           'frequencies': self.df.dtypes==False})
        return info


    def category(self,columns):
        """
        """
        self.metadata['category']=self.metadata.index.isin(columns) | self.metadata['category']==True
        self.nb_category=self.metadata[self.metadata['category']==True].count()['category']
        self.category_list=list(self.metadata[self.metadata['category']==True].reset_index()['index'])

    def continuous(self,columns):
        """
        """
        self.metadata['continuous']=self.metadata.index.isin(columns) | self.metadata['continuous']==True
        self.nb_category=self.metadata[self.metadata['continuous']==True].count()['continuous']

    def clusters(self,clusters):
        """set the number of clusters in the dataset
        """
        self.nb_clusters=clusters

    def targets(self,targets):
        """
        """
        # if not isinlist(self._targets,self._metadata['columns']):
        #     self._mdtargets=self._metadata[self._metadata['targets']==True]['columns'][0]
        # else:
        #     self._mdtargets=self._targets
        # self._metadata['targets']=self._metadata['columns']==self._mdtargets

        if targets is None or len(targets)==0 or targets not in self.df.columns:
            self.targets_column=''
            self.datatype_targets=None
            self.distinct_targets=None
            self.distratio_targets=None
            self.max_targets=None
            self.min_targets=None
            self.most_frequent_targets=None
            self.less_frequent_targets=None
            self.median_frequent_targets=None
            self.mean_targets=None
            self.stdev_targets=None
            self.q10_targets = None
            self.q25_targets = None
            self.q50_targets = None
            self.q75_targets = None
            self.q90_targets = None
        else:
            self.targets_column=targets
            self.datatype_targets=str(self.df.dtypes[targets])
            self.distinct_targets=self.metadata.loc[targets,'distinct']
            self.distratio_targets=self.metadata.loc[targets,'distratio']
            self.max_targets=self.df.max()[targets]
            self.min_targets=self.df.min()[targets]
            self.most_frequent_targets=list(self.df[targets].value_counts().head(1).reset_index()['index'])[0]
            self.less_frequent_targets=list(self.df[targets].value_counts().tail(1).reset_index()['index'])[0]
            self.median_frequent_targets=self.df[targets].iloc[int(len(self.df[targets])/2)]
            if self.df[targets].dtypes.kind in ['i','f']:
                self.mean_targets= self.df.mean()[targets]
                self.stdev_targets=np.sqrt(self.df.var())[targets]
                self.q10_targets = self.df.quantile(0.10,interpolation='lower')[targets]
                self.q25_targets = self.df.quantile(0.25,interpolation='lower')[targets]
                self.q50_targets = self.df.quantile(0.5,interpolation='midpoint')[targets]
                self.q75_targets = self.df.quantile(0.75,interpolation='higher')[targets]
                self.q90_targets#import ipdb; ipdb.set_trace() = self.df.quantile(0.90,interpolation='higher')[targets]
            self.metadata['targets']=self.metadata.index==targets


    def transform(self,columns):
        """
        """
        self.metadata['transform']=self.metadata.index.isin(columns) | self.metadata['transform']==True

    def rename(self,columns):
        """
        """
        self.metadata['transform']=self.metadata.index.isin(columns) | self.metadata['transform']==True


    def drop(self,columns):
        """
        """
        self.metadata['drop']=self.metadata.index.isin(columns) | self.metadata['drop']==True
        self.drop_list=list(self.metadata[self.metadata['drop']==True].reset_index()['index'])


    def dummies(self,columns):
        """
        """
        self.metadata['dummies']=self.metadata.index.isin(columns) | self.metadata['dummies']==True
        self.dummies_list=list(self.metadata[self.metadata['dummies']==True].reset_index()['index'])

    def ordinals(self,column):
        """
        """
        self.metadata['ordinals']=self.metadata.index.isin([column]) | self.metadata['ordinals']==True
        self.ordinals_list=list(self.metadata[self.metadata['ordinals']==True].reset_index()['index'])


    def frequencies(self,columns):
        """
        """
        self.metadata['frequencies']=self.metadata.index.isin(columns) | self.metadata['frequencies']==True
        self.frequencies_list=list(self.metadata[self.metadata['frequencies']==True].reset_index()['index'])

    def _set_algo(self,algo):
        """
        """
        self._algo=algo

    def _get_algo(self):
        """
        """
        return self._algo

    algo=property(_get_algo,_set_algo)

    def suggested_param(self):
        """
        """
        #self._metadata['drop']=self._metadata['columns'].isin(columns) | self._metadata['drop']==True
        return {'targets':self.targets_column,'drop':self.drop_list,'category':self.category_list,'ordinals':self.ordinals_list,        'frequencies':self.frequencies_list,'dummies':self.dummies_list,'algo':self.algo,'clusters':self.nb_clusters}
