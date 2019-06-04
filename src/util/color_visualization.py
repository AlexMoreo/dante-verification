from matplotlib.cm import get_cmap

def color_tag(index, text, probability, cmap):
    probability *= 0.6
    # probability = (probability-0.5)*0.75+0.5
    r,g,b,_ = cmap(probability)
    # reliable = abs(probability-0.5) > 0.25*0.75
    # text = '<font color="white">{}</font>'.format(text) if reliable else text
    return f'<b>&nbsp;P{index}:</b> <a style="background-color:rgb({r*255:.0f},{g*255:.0f},{b*255:.0f});">{text} </a>'

def color(path, texts, probabilities, title, paragraph_offset=1):
    # cmap = get_cmap('RdYlGn')
    # cmap = get_cmap('Greens')
    cmap = get_cmap('Greys')

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
        for i,(line,probability) in enumerate(zip(texts,probabilities)):
            fo.write(color_tag(paragraph_offset + i, line,probability,cmap))
        fo.write("""
                </body>
                </html>
                """)

