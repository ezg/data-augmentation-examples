import pandas as pd
import os
import re


with open('expenses_raw.csv') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

with open('expenses_clean.csv', 'w') as out:
    h = ['UNITID', 'Name', 'Years', 'Enrollment', 'ExpensesTotal', 'ExpensesPerStudent']
    out.write(', '.join(h) + '\n')
    for l in content:
        newL = []
        t1 = l.split(' ')
        if t1[0].isdigit():
            newL.append(t1[0])
            delimiters = ['4-Year', '2-Year']
            regexPattern = '|'.join(map(re.escape, delimiters))
            t2 = re.split(regexPattern, " ".join(t1[1:]))
            t3 = t2[-1].split(' ')
            if len(t3) == 3 and t3[1].isdigit() and t3[2].isdigit():
                newL.append(t2[0].strip().replace(',', '')) # name
                newL.append('2' if '2-Year' in l else '4')
                newL.append(t3[1].strip())
                newL.append(t3[2].strip())
                newL.append(float(t3[2].strip()) / float(t3[1].strip()))
                out.write(', '.join([str(x) for x in newL]) + '\n')
                print(newL)   