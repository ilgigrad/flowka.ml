from random import randint,sample

class fcol:
    '''give some color themes for Flowka
    '''
    _dictcolors={'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azure': '#F0FFFF',
    'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000', 'blanchedalmond': '#FFEBCD', 'blue': '#0000FF',
    'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00',
    'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED', 'cornsilk': '#FFF8DC', 'crimson': '#DC143C',
    'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9',
    'darkgreen': '#006400', 'darkgrey': '#A9A9A9', 'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F', 'darkslategrey': '#2F4F4F', 'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dimgrey': '#696969',
    'dodgerblue': '#1E90FF', 'firebrick': '#B22222', 'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080',
    'green': '#008000', 'greenyellow': '#ADFF2F', 'grey': '#808080', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4',
    'indianred': '#CD5C5C', 'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgray': '#D3D3D3', 'lightgreen': '#90EE90',
    'lightgrey': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA',
    'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0',
    'lime': '#00FF00', 'limegreen': '#32CD32', 'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000',
    'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585', 'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD', 'navy': '#000080', 'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23',
    'orange': '#FFA500', 'orangered': '#FF4500', 'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9', 'peru': '#CD853F',
    'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080', 'rebeccapurple': '#663399',
    'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513', 'salmon': '#FA8072',
    'sandybrown': '#F4A460', 'seagreen': '#2E8B57', 'seashell': '#FFF5EE', 'sienna': '#A0522D', 'silver': '#C0C0C0',
    'skyblue': '#87CEEB', 'slateblue': '#6A5ACD', 'slategray': '#708090', 'slategrey': '#708090', 'snow': '#FFFAFA',
    'springgreen': '#00FF7F', 'steelblue': '#4682B4', 'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8',
    'tomato': '#FF6347', 'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00', 'yellowgreen': '#9ACD32',
    'prune': '#615375', 'peach': '#E5625C', 'apricot': '#F9BF76', 'electric': '#34A9D4', 'rose': '#FED4E0',
    'prune': '#FED4E0', 'anis': '#DEDD23', 'sadsea': '#72B2C5', 'blood': '#D1313D', 'darkflowka': '#202020'}
    _listpalettes=['hot', 'cool', 'viridis', 'magma', 'spectral', 'Greens',
    'Reds', 'Blues', 'Greys', 'Oranges', 'pink', 'copper', 'gnuplot', 'BuGn', 'YlGn',
    'YlGnBu', 'BuPu', 'RdPu', 'YlOrBr', 'OrRd', 'YlOrRd', 'binary', 'spectral',
    'spring', 'summer', 'autumn', 'winter', 'bone', 'GnBu', 'PuBu', 'PuBuGn']
    _listcmaps=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
    'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
    'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
    'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
    'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    'Spectral', 'Spectral_r', 'Vega10', 'Vega10_r', 'Vega20', 'Vega20_r', 'Vega20b', 'Vega20b_r', 'Vega20c',
    'Vega20c_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd',
    'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
    'brg_r', 'bwr', 'bwr_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
    'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
    'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
    'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r',
    'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r',
    'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
    'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spectral', 'spectral_r',
    'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
    'tab20c_r', 'terrain', 'terrain_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

    def __init__(self):
        #super(fpd,self).__init__()
        #self._data = df._data
        self.flowkapalette=[self._dictcolors['prune'],self._dictcolors['blood'],self._dictcolors['peach'],
        self._dictcolors['apricot'],self._dictcolors['rose'],self._dictcolors['sadsea'],self._dictcolors['electric'],
        self._dictcolors['anis']]
        self._cmap=-1
        self._palette=-1
        self._color=-1


    def __set_palette(self,numpalette=None):
        '''set a color palette
           numpalette : num of palette to set
           numpalette = -1 -> display all available color palettes
           numpalette = None -> set a palette randomly
        '''
        self._palette=-1
        if numpalette in range(len(self._listpalettes)):
            self._palette=numpalette
        elif numpalette is not None:
            for id ,palette in enumerate(self._listpalettes):
                print ('{0}:\t{1}'.format(id,palette))

    def __get_palette(self):
        '''return the selected palette
        '''
        if self._palette==-1:
            return self._listpalettes[randint(0,len(self._listpalettes)-1)]
        else:
            return self._listpalettes[self._palette]

    palette = property(__get_palette,__set_palette)


    def __set_cmap(self,numcmap=None):
        '''set a color map
           numcmap : num of the cmap to set
           numcmap = -1 -> display all available color maps
           nucmap = None -> set a cmap randomly
        '''
        self._cmap= -1
        if numcmap in range(len(self._listcmaps)):
            self._cmap=numcmap
        elif numcmap is not None:
            for id ,cmap in enumerate(self._listcmaps):
                print ('{0}:\t{1}'.format(id,cmap))


    def __get_cmap(self):
        '''return the selected cmap
        '''
        if self._cmap==-1:
            return self._listcmaps[randint(0,len(self._listcmaps)-1)]
        else:
            return self._listcmaps[self._cmap]

    cmap = property(__get_cmap,__set_cmap)

    def __set_color(self,codecolor=None):
        ''' set a color
            codecolor : name of the color to display
            codecolor = None return a randomly a flowka color
        '''
        self._color=-1
        if codecolor in self._dictcolors:
            self._color=codecolor
        elif codecolor is not None:
            for id ,color in enumerate(self._dictcolors.items()):
                print ('{0}:\t{1}'.format(id,color))

    def __get_color(self):
        ''' return the selected color
        '''
        if self._color==-1:
            return self.flowkapalette[randint(0,len(self.flowkapalette)-1)]
        else:
            return self._dictcolors[self._color]

    color = property(__get_color,__set_color)

    def multicolor(self,nbcolor=2):
        ''' return a list of random color in flowka palette
            multicolor(3) -> ['#D1313D', '#FED4E0', '#72B2C5']
        '''
        colorset=set()
        if nbcolor>len(self.flowkapalette)-1:
            print("too many colors, max is : {0}".format(len(self.flowkapalette)-1))
        else:
            while len(colorset)<nbcolor:
                colorset.add(self.flowkapalette[randint(0, len(self.flowkapalette)-1)])
        return list(colorset)

    def c(self,color=None):
        """give a single selected color from name
           None return darkgrey
        """
        if color in self._dictcolors:
            return self._dictcolors[color]
        else:
            return self._dictcolors['darkflowka']
