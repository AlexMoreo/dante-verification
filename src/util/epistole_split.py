import os

dir = '../../testi_1'
author = 'Misc'
file=author+'_Epistole.txt'



order = 0
epistola=[]
for line in open(os.path.join(dir,file), 'rt').readlines():
    line = line.strip()
    if line:
        epistola.append(line)
    else:
        epistola = '\n'.join(epistola)
        open(os.path.join(dir,'{}_epistola{}.txt'.format(author,order)), 'wt').write(epistola)
        order += 1
        epistola = []

if epistola:
    epistola = '\n'.join(epistola)
    open(os.path.join(dir, '{}_epistola{}.txt'.format(author,order)), 'wt').write(epistola)


