
from flowkafunc import nvl
from flowkapd import fpd
from flowkamo import fmo


def predictsave(dataframe,name, savefile=None):
    """give prediction for some values set manually
    """
    #predictor=fpd(pd.read_csv("flowka_Advertising_preds.csv"),'advertising')
    #print('load dataset')
    _fpdpr=fpd(dataframe,name)
    _fpdpr._talk.quiet
    #print('clean data')
    _fpdpr.batch()
    #print('load model')
    _fmopr=fmo(_fpdpr)
    _fmopr.load_model()

    #print('prediction')
    _fmopr.predict()
    #print('restore dataset values')
    _fpdpr.restore_df()
    #print('save model')
    _fpdpr.add=_fmopr.pred_append(_fmopr.algo)
    _fpdpr.save(pdfile=nvl(savefile,'./work/pred_'+_fpdpr.name),mode='csv')

    return _fpdpr.df
