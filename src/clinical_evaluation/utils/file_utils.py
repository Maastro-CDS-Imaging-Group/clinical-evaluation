import pandas as pd
import itertools    

def convert_to_multilevel_df(df, sep='.'):
    indices = set()
    groups = set()
    others = set()
    for c in df.columns:
        if sep in c:
            (i, g) = c.split(sep)
            c2 = pd.MultiIndex.from_tuples((i, g),)
            indices.add(i)
            groups.add(g)
        else:
            others.add(c)
    columns = list(itertools.product(indices, groups))
    columns = pd.MultiIndex.from_tuples(columns)
    ret = pd.DataFrame(columns=columns)
    for c in columns:
        ret[c] = df['%s%s%s' % (c[0], sep, c[1])]
    for c in others:
        ret[c] = df['%s' % c]
    ret.rename(columns={'total': 'total_indices'}, inplace=True)

    return ret