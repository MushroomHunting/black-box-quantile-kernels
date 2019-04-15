# https://github.com/BIDS/colormap/blob/master/colormaps.py

# New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
# and (in the case of viridis) Eric Firing.
#
# This file and the colormaps in it are released under the CC0 license /
# public domain dedication. We would appreciate credit if you use or
# redistribute these colormaps, but do not impose any legal restrictions.
#
# To the extent possible under law, the persons who associated CC0 with
# mpl-colormaps have waived all copyright and related or neighboring rights
# to mpl-colormaps.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.



# Dictionary of all the standard named colours
# Generated from:
#
# import matplotlib
# for name, hex in matplotlib.colors.cnames.iteritems():
#     print(name, hex)
#
# ref: http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib

named_colours = {'yellowgreen': '#9ACD32', 'aquamarine': '#7FFFD4', 'maroon': '#800000', 'cyan': '#00FFFF',
                 'palevioletred': '#DB7093', 'brown': '#A52A2A', 'darkolivegreen': '#556B2F',
                 'darkslategray': '#2F4F4F', 'darkgray': '#A9A9A9', 'lightcyan': '#E0FFFF', 'lawngreen': '#7CFC00',
                 'sage': '#87AE73', 'darkcyan': '#008B8B', 'darkorchid': '#9932CC', 'oldlace': '#FDF5E6',
                 'lemonchiffon': '#FFFACD', 'dimgray': '#696969', 'palegoldenrod': '#EEE8AA', 'greenyellow': '#ADFF2F',
                 'goldenrod': '#DAA520', 'navy': '#000080', 'mediumaquamarine': '#66CDAA', 'skyblue': '#87CEEB',
                 'midnightblue': '#191970', 'tomato': '#FF6347', 'lavender': '#E6E6FA', 'gainsboro': '#DCDCDC',
                 'palegreen': '#98FB98', 'lightseagreen': '#20B2AA', 'lightsteelblue': '#B0C4DE', 'black': '#000000',
                 'azure': '#F0FFFF', 'slategrey': '#708090', 'darkgoldenrod': '#B8860B', 'orange': '#FFA500',
                 'springgreen': '#00FF7F', 'firebrick': '#B22222', 'gray': '#808080', 'purple': '#800080',
                 'deepskyblue': '#00BFFF', 'darkgreen': '#006400', 'plum': '#DDA0DD', 'mediumorchid': '#BA55D3',
                 'bisque': '#FFE4C4', 'green': '#008000', 'darksalmon': '#E9967A', 'gold': '#FFD700',
                 'whitesmoke': '#F5F5F5', 'olive': '#808000', 'indianred': '#CD5C5C', 'darkblue': '#00008B',
                 'papayawhip': '#FFEFD5', 'mintcream': '#F5FFFA', 'dodgerblue': '#1E90FF', 'mistyrose': '#FFE4E1',
                 'steelblue': '#4682B4', 'salmon': '#FA8072', 'beige': '#F5F5DC', 'darksage': '#598556',
                 'mediumseagreen': '#3CB371', 'aqua': '#00FFFF', 'lime': '#00FF00', 'seagreen': '#2E8B57',
                 'chocolate': '#D2691E', 'ghostwhite': '#F8F8FF', 'yellow': '#FFFF00', 'lightslategrey': '#778899',
                 'thistle': '#D8BFD8', 'lightgrey': '#D3D3D3', 'forestgreen': '#228B22', 'orchid': '#DA70D6',
                 'red': '#FF0000', 'dimgrey': '#696969', 'paleturquoise': '#AFEEEE', 'fuchsia': '#FF00FF',
                 'lightpink': '#FFB6C1', 'mediumturquoise': '#48D1CC', 'moccasin': '#FFE4B5', 'linen': '#FAF0E6',
                 'darkgrey': '#A9A9A9', 'darkslateblue': '#483D8B', 'white': '#FFFFFF', 'mediumspringgreen': '#00FA9A',
                 'mediumblue': '#0000CD', 'peru': '#CD853F', 'sienna': '#A0522D', 'chartreuse': '#7FFF00',
                 'darkviolet': '#9400D3', 'darkturquoise': '#00CED1', 'lightgreen': '#90EE90',
                 'lightgoldenrodyellow': '#FAFAD2', 'antiquewhite': '#FAEBD7', 'cornflowerblue': '#6495ED',
                 'snow': '#FFFAFA', 'lightslategray': '#778899', 'mediumslateblue': '#7B68EE', 'tan': '#D2B48C',
                 'seashell': '#FFF5EE', 'lightcoral': '#F08080', 'mediumvioletred': '#C71585', 'crimson': '#DC143C',
                 'slategray': '#708090', 'saddlebrown': '#8B4513', 'blue': '#0000FF', 'deeppink': '#FF1493',
                 'navajowhite': '#FFDEAD', 'lightsage': '#BCECAC', 'pink': '#FFC0CB', 'cornsilk': '#FFF8DC',
                 'ivory': '#FFFFF0', 'grey': '#808080', 'mediumpurple': '#9370DB', 'darkseagreen': '#8FBC8F',
                 'turquoise': '#40E0D0', 'hotpink': '#FF69B4', 'cadetblue': '#5F9EA0', 'sandybrown': '#FAA460',
                 'teal': '#008080', 'blueviolet': '#8A2BE2', 'indigo': '#4B0082', 'lightsalmon': '#FFA07A',
                 'khaki': '#F0E68C', 'violet': '#EE82EE', 'darkslategrey': '#2F4F4F', 'rosybrown': '#BC8F8F',
                 'olivedrab': '#6B8E23', 'aliceblue': '#F0F8FF', 'blanchedalmond': '#FFEBCD', 'honeydew': '#F0FFF0',
                 'floralwhite': '#FFFAF0', 'magenta': '#FF00FF', 'slateblue': '#6A5ACD', 'lavenderblush': '#FFF0F5',
                 'lightskyblue': '#87CEFA', 'darkred': '#8B0000', 'darkorange': '#FF8C00', 'lightgray': '#D3D3D3',
                 'limegreen': '#32CD32', 'darkkhaki': '#BDB76B', 'coral': '#FF7F50', 'lightblue': '#ADD8E6',
                 'wheat': '#F5DEB3', 'orangered': '#FF4500', 'darkmagenta': '#8B008B', 'burlywood': '#DEB887',
                 'peachpuff': '#FFDAB9', 'powderblue': '#B0E0E6', 'silver': '#C0C0C0', 'lightyellow': '#FFFFE0',
                 'royalblue': '#4169E1'}

named_CB = {"blue": '#377eb8', "orange": '#ff7f00', "green": '#4daf4a',
            "pink": '#f781bf', "brown": '#a65628', "purple": '#984ea3',
            "darkgrey": '#999999', "red": '#e41a1c', "limeyellow": '#dede00',
            "purple_medium": "#9b59b6", "blue_medium": "#3498db",
            "teal_light": "#95a5a6", "red_light": "#e74c3c", "navygrey_dark": "#34495e",
            "green_light": "#2ecc71"}

