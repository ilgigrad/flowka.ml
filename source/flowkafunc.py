

class talk:
    "print information while running"

    def __init__(self,loud=None):
        self._loud=nvl(loud,True) #by dafault talk is active

    def talk(self,string):
        if self._loud:
            print(string)

    def _set_loud(self,loud=True):
        self._loud= loud==True
        return self._loud

    loud = property(_set_loud)

    def _set_quiet(self,quiet=True):
        self._loud= quiet==False
        return self._loud

    quiet = property(_set_quiet)

def removelist(sourcelist,removelist):
    return [x for x in sourcelist if x not in removelist]

def isinlist (includedlist,containlist):
    """return if a list is included in other list
    """
    if nvl(includedlist,None) is None:
        return False
    elif type(includedlist) is str:
        includedlist=[includedlist]
    else:
        includedlist=list(includedlist)
    if nvl(containlist,None) is None:
        return False
    elif type(containlist) is str:
        containlist=[containlist]
    elif type(containlist) is not list:
        icontainlist=list(containlist)

    for element in includedlist:
        if element not in list(containlist):
            return False
    return True

def isfloat(value):
    """return if a value is a float
    """
    try:
        float(str(value))
        return True
    except ValueError:
        return False

def unionlist (*alist):
    """union of two lists
       a=[foo,bar,banana]
       b=[foo,apple,banana]
       unionlist(a,b) -> [foo,bar,banana,apple]
    """
    lists=[]
    for xlist in (alist):
        if xlist is None:
            lists.append([])
        elif type(xlist)!=list:
            lists.append([xlist])
        else:
            lists.append(xlist)
    return list(set(lists[0]).union(*lists[1:]))

def intersectionlist (*alist):
    """intersection of two lists
       a=[foo,bar,banana]
       b=[foo,apple,banana]
       intesectionlist(a,b) -> [foo,banana]
    """
    lists=[]
    for xlist in alist:
        if xlist is None:
            lists.append([])
        elif type(xlist)!=list:
            lists.append([xlist])
        else:
            lists.append(xlist)
    return list(set(lists[0]).intersection(*lists[1:]))

def nvl(var, val):
  if var is None or (type(var) not in [bool,int,float] and len(var)==0):
    return val
  return var

def nnone(var):
  if var is None:
    return 'None'
  return var

def slugify(s):
    import re
    s = s.lower()
    # "[some] _ article's_title--"
    # "[some]___article's_title__"
    for c in [' ', '-', '.', '/']:
        s = s.replace(c, '_')

    # "[some]___article's_title__"
    # "some___articles_title__"
    s = re.sub('\W', '', s)

    # "some___articles_title__"
    # "some   articles title  "
    s = s.replace('_', ' ')

    # "some   articles title  "
    # "some articles title "
    s = re.sub('\s+', ' ', s)

    # "some articles title "
    # "some articles title"
    s = s.strip()

    # "some articles title"
    # "some-articles-title"
    s = s.replace(' ', '_')

    return s
