from matplotlib.cm import get_cmap

def color_tag(text, probability, cmap):
    r,g,b,_ = cmap(1-probability)
    return '<a style="background-color:rgb({:.0f},{:.0f},{:.0f});">{} </a>'.format(r*255,g*255,b*255,text)

def color(path, texts, probabilities, title):
    cmap = get_cmap('RdYlGn')

    with open(path, 'wt') as fo:
        fo.write("""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>{}</title>
                </head>
                <body>
                <h1>{}</h1>
                """.format(title,title))
        fo.write('<p>')
        for line,probability in zip(texts,probabilities):
            fo.write(color_tag(line,probability,cmap))
        fo.write('</p>')
        fo.write("""
                </body>
                </html>
                """)

