import pandas as pd
import numpy as np
from core.utils import NANS
from storage.storage import Actual_Storage


class DataSet():

    _df=pd.DataFrame()
    _is_valid=True
    def __init__(self,df=None):
        if df is not None:
            self._df=df
            self.find_NaN()

    def read(self,file):
        #import ipdb; ipdb.set_trace()
        ext=file.split('.')[-1].lower()
        file=Actual_Storage().get(file)
        if ext=='csv':
            try:
                self._df=pd.read_csv(file,low_memory=False)
            except:
                self._is_valid=False
        elif ext=='html': #may have many tables  -> select the good one #df is a list of DataFFrame (1 for each sheet)
            try:
                self._df=pd.read_html(file,header=0,low_memory=False)
            except:
                self._is_valid=False
        elif ext in ['xls','xlsx']: #may have many sheets, select the good one
            try:
                self._df=pd.read_excel(file,low_memory=False) #df is a dict of DataFFrame (1 for each sheet)
            except:
                self._is_valid=False
        elif ext =='json': #may have many sheets, select the good one
            try:
                self._df=pd.read_json(file,low_memory=False)
            except:
                self._is_valid=False

        elif ext in ['txt', 'log']: #may have many sheets, select the good one
            try:
                self._df=pd.read_table(file,low_memory=False)

            except:
                try:
                    self._df=pandas.read_fwf(file, colspecs='infer', widths=None,low_memory=False)
                except:
                    self._is_valid=False
        else:
            self._is_valid=False

        if self._is_valid:
            self.find_NaN()

        return self._is_valid

    def find_NaN(self):
        for col in self.columns:
            self.df[col]=self.df[col].apply(lambda x : np.NaN if x is None or x in NANS else x)

    def sample(self,nrows=5,random=False):
        if self._df.size==0:
            return False
        elif random:
            return self._df.sample(nrows)
        elif nrows>0:
            return self._df.head(nrows)
        else:
            return self._df.tail(-nrows)

    @property
    def df(self):
        return self._df

    @property
    def details(self):
        """
        give some detailed informations about
        distinct or nan values in columns
        """

        def simpletype(x):
            typeDict={
                'flo':'decimal',
                'com':'decimal',
                'int':'integer',
                'uin':'integer',
                'uns':'integer',
                'obj':'string',
                'char':'string',
                'byt':'string',
                'str':'string',
                'boo':'binary',
                'dat':'date',
                'tim':'time',
                }

            if x is None:
                return 'no-type'
            else:
                return typeDict.get(str(x)[:3],str(x))

        def length(x):
            if x is None or x==np.NaN:
                return 0
            else:
                try:
                    return len(x)
                except:
                    return len(str(x))

        if self._df.size>0:
            info={}
            info['type']= self.df.dtypes.apply(lambda x: simpletype(x))
            info['min']= self.df.min()
            info['max']= self.df.max()
            #compute max and min length of column
            cols={}
            df=pd.DataFrame()

            for c in self.columns:
                cols[c]=self._df[c].apply(lambda x: length(x))
                df=pd.concat([df,cols[c]],axis=1,)

            info['distinct']= self.df.nunique()
            info['nan']= self.df.isnull().sum()
            info['valid']= self.df.count()
            info['count']= info['valid']+info['nan']
            info['maxlen']=df.max()
            info['minlen']=df.min()
            #--------------------------------------
            info['mean']= self.df.mean().apply(lambda x : np.round(x,2))
            info['stddev']= np.sqrt(self.df.var()).apply(lambda x : np.round(x,2))

            self.df['forstatsonly']=pd.Series(np.zeros(len(df)))
            info['q10']= self.df.quantile(0.10,interpolation='lower').apply(lambda x : np.round(x,2))
            info['q25']= self.df.quantile(0.25,interpolation='lower').apply(lambda x : np.round(x,2))
            info['q50']= self.df.quantile(0.5,interpolation='midpoint').apply(lambda x : np.round(x,2))
            info['q75']= self.df.quantile(0.75,interpolation='higher').apply(lambda x : np.round(x,2))
            info['q90']= self.df.quantile(0.90,interpolation='higher').apply(lambda x : np.round(x,2))
            self.df.drop(['forstatsonly'],axis=1,inplace=True)
            #info['ratio']=distratio.apply(lambda x : np.round(x,2))
            info=pd.DataFrame(info)
            info.drop(['forstatsonly'],axis=0,inplace=True)
            info['maxlen']=info['maxlen'].astype(int,inplace=True)
            info['minlen']=info['maxlen'].astype(int,inplace=True)
            info['nan']=info['nan'].astype(int,inplace=True)
            info['count']=info['count'].astype(int,inplace=True)
            info['valid']=info['valid'].astype(int,inplace=True)
            info['distinct']=info['distinct'].astype(int,inplace=True)
            #distratio = info['count']/info['distinct']
            #validratio= info['count']/info['valid']
            #nanratio= info['count']/info['nan']
            cols=['type','count','distinct','nan','valid','max','min','mean','stddev','q10','q25','q50','q75','q90','maxlen','minlen']
            return info[cols]

        return False

    @property
    def columns(self):
        """
        return list of columns in Dataset
        """
        return list(self._df.columns)

    @property
    def shape(self):
        """
        return shape of Dataset
        """
        return self._df.shape

    @property
    def ncols(self):
        """
        return number of columns
        """
        return self._df.shape[1]

    @property
    def nrows(self):
        """
        return number of rows
        """
        return self._df.shape[0]

    def snippet(self,nrows=100,headtail=10):
        """ return a snippet of the Dataset
            head and tail sizes are given by headtail argument
            the other rows are extracted from the dataframe with a random key
        """
        if nrows>self.nrows:
            nrows=self.nrows
        if nrows<2*headtail:
            headtail=int(nrows/2)
        body=nrows-(2*headtail)
        sn=self._df.head(headtail).append(self._df[headtail:-headtail].sample(body)).append(self._df.tail(headtail)).sort_index().drop_duplicates()
        return sn.transpose()
