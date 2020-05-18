import numpy as np
import pandas as pd
import re
from io import StringIO
import json
import gzip

import memolon.src.constants as cs



def list_intersection(list1, list2):
    return sorted(list(set(list1).intersection(set(list2))))


def list_union(list1, list2):
    return sorted(set(list1).union(set(list2)))


def get_language_table():
    table = pd.read_csv(cs.language_table, sep='\t', encoding='utf-8', index_col='iso')
    return table


# Load language table and mappings between iso codes and full names
language_table = get_language_table()
iso_2_fullname = {iso:language_table.loc[iso, "google_fullname"] for iso in language_table.index}
fullname_2_iso = {value: key for key, value in iso_2_fullname.items()}


def get_source(split):
    assert split in ['train', 'dev', 'test', 'full']

    VAD = pd.read_csv(cs.warriner_vad, encoding='utf-8', index_col=1, keep_default_na=False)
    VAD = VAD.rename(columns={"V.Mean.Sum": "valence", "A.Mean.Sum": "arousal", "D.Mean.Sum": "dominance"})
    VAD = VAD[["valence", "arousal", "dominance"]]

    BE = pd.read_csv(cs.warriner_be, sep='\t', encoding='utf-8', index_col=0, keep_default_na=False)
    BE = BE.rename(columns={c:c.lower() for c in BE.columns})
    df = pd.concat([VAD, BE], join="inner", axis=1)
    if split == "full":
        words = get_split_words("train") + get_split_words("dev") + get_split_words("test")
        df = df.loc[words]
    else:
        df = df.loc[get_split_words(split)]
    return df


def get_TargetMT(iso, split="full"):
    assert split in ["train", "dev", "test", "full"]
    source = get_source(split)
    translation = get_translation_dict(iso)
    # change index names according to translation, leave values unchanged
    targetMT = source.rename(index=translation)
    return targetMT


def get_TargetPred(iso, split="full", version="grouped"):
    """

    :param iso: iso language code
    :param split: one of "train", "dev", "test" or "full"
    :param version:
        grouped: Multi-Task learning between VAD and BE5 respectively but not between both groups
        ridge: Using ridge regression baseline
        stl: using single task learning for everything
        mtl: Multi-Task learning across all variables
        vad_mtl_be5_stl: VAD with MTL but BE5 with STL
        vad_stl_be5_mtl: VAD with STL but BE5 with MTL
    :return: TargetPred lexicon as a pandas.DataFrame
    """
    if version == "grouped":
        full = pd.read_csv(cs.TARGETPRED_MTL_GROUPED / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)

    elif version == "ridge":
        full = pd.read_csv(cs.TARGETPRED_RIDGE / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)

    elif version == "stl":
        full = pd.read_csv(cs.TARGETPRED_STL / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)

    elif version == "mtl":
        full = pd.read_csv(cs.TARGETPRED_MTL_all  / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)

    elif version == "vad_mtl_be5_stl":
        vad = pd.read_csv(cs.TARGETPRED_MTL_GROUPED / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)[cs.vad]
        be5 = pd.read_csv(cs.TARGETPRED_STL / f"{iso}.tsv",
                           sep='\t',
                           index_col=0,
                           keep_default_na=False)[cs.be5]
        full = pd.concat([vad, be5], axis="columns")

    elif version == "vad_stl_be5_mtl":
        vad = pd.read_csv(cs.TARGETPRED_STL / f"{iso}.tsv",
                          sep='\t',
                          index_col=0,
                           keep_default_na=False)[cs.vad]
        be5 = pd.read_csv(cs.TARGETPRED_MTL_GROUPED / f"{iso}.tsv",
                          sep='\t',
                          index_col=0,
                           keep_default_na=False)[cs.be5]
        full = pd.concat([vad, be5], axis="columns")

    else:
        raise NotImplementedError(f"{version} is no valid version identifier.")

    translation = get_translation_dict(iso)

    if split == "full":
        return full

    elif split == "train":
        source_train_words = get_split_words("train")
        target_train_words = set([translation[w] for w in source_train_words])
        target_train = full.loc[target_train_words]
        return target_train

    elif split == "dev":
        source_train_words = get_split_words("train")
        source_dev_words = get_split_words("dev")

        # remove train entries from dev
        target_train_words = [translation[w] for w in source_train_words]
        target_dev_words = [translation[w] for w in source_dev_words]
        target_dev_words = set(target_dev_words).difference(set(target_train_words))

        target_dev = full.loc[target_dev_words]

        return target_dev

    elif split == "test":
        source_train_words = get_split_words("train")
        source_dev_words = get_split_words("dev")
        target_train_words = [translation[w] for w in source_train_words]
        target_dev_words = [translation[w] for w in source_dev_words]

        # too slow
        #target_test_words = set(full.index).difference(set(target_train_words)).difference(set(target_dev_words))
        #target_test = full.loc[target_test_words]
        # equivalent but faster:
        target_test = full.drop(index=list_union(target_train_words, target_dev_words), errors="ignore")

        return target_test

    else:
        raise NotImplementedError


def get_split_words(split):
    with open(cs.SOURCE / '{}.txt'.format(split), 'r') as f:
        words = f.read().split('\n')
        return words


def get_translation_dict(iso):
    p = cs.TRANSLATION / f"{iso}.json"
    with open(p) as f:
        dc = json.load(f)
    return dc


def load_vectors(path=None, iso=None, limit=None):
    """
    :param iso: iso language code ("en", "de", "tr", ...)
    :param path: path to *.gz file
    :param limit: maximum number of lines/tokens
    :return: dict
    """
    if path==None and iso==None:
        raise ValueError("You need to provide 'path=' or 'iso='")
    if path!=None and iso!=None:
        raise ValueError("You should not provide both 'path=' and 'iso='")
    elif iso:
        path = cs.EMBEDDINGS / 'cc.{}.300.vec.gz'.format(iso)

    print("Loading embeddings ...")
    f = gzip.open(path, 'rb')
    n, d = map(int, f.readline().split())
    data = {}
    counter = 0
    for line in f:
        line = line.decode()
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
        if limit:
            if counter >= limit:
                break
            else:
                counter += 1
    f.close()
    return data


def get_features(entry, embs):
    vector = np.zeros(300)
    if entry in embs:
        vector = embs.get(entry)
    else:
        substrings = re.split(" |'|-", entry)
        counter = 0
        for s in substrings:
            if s in embs:
                vector += embs.get(s)
                counter +=1
        if counter !=0:
            vector = vector/counter
    return vector


def postprocess_lexicon(iso, target_pred):

    # the aim here is: sorting TargetPred on the following way:
    # first the original entries in TargetMT, then the remaining words from embedding model keeping their order

    target_mt = get_TargetMT(iso=iso, split="full")
    translated_words = target_mt.index.tolist()

    # get the words from the embeddings (keeping the original order)
    f = gzip.open(cs.EMBEDDINGS / 'cc.{}.300.vec.gz'.format(iso), 'rb')
    embs_words = []
    for line in f:
        line = line.decode()
        token = line.rstrip().split(' ')[0]
        embs_words.append(token)
    embs_words = embs_words[1:]  # get rid of first line in Embedding files (meta information)
    f.close()

    total = translated_words + embs_words
    total.append('<UNK>')  # total is concatenation, so it includes duplicates

    target_pred.reset_index(inplace=True)
    target_pred.drop_duplicates(subset='word', inplace=True)
    target_pred.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)  # remove rows with empty entries ("")
    target_pred.set_index('word', inplace=True)

    sorted = target_pred.reindex(total)  # unfortunately duplicates return
    sorted.reset_index(inplace=True)
    sorted.drop_duplicates(subset='word', inplace=True)  # drop duplicates again
    sorted.set_index('word', inplace=True)

    return sorted


def get_en1_gold():
    df = get_source('test')[["valence", "arousal", "dominance"]]
    return df

def get_en2_gold():
    anew = pd.read_csv(cs.en2, sep = '\t')
    anew.columns = ['word', 'valence', 'arousal', 'dominance']
    anew.set_index('word', inplace = True)
    #drop duplicates
    anew = anew[~anew.index.duplicated()]
    return anew

def get_en3_gold():
    be = pd.read_excel(cs.en3, index_col=0)
    rename= {
    'mean_hap': 'joy',
    'mean_ang': 'anger',
    'mean_sad': 'sadness',
    'mean_fear': 'fear',
    'mean_dis': 'disgust'
    }

    be = be[~be.index.isna()]
    be = be.rename(columns=rename)
    for c in be.columns:
        if c not in rename.values():
            be.drop(columns=c, inplace=True)

    be.index = [x.strip() for x in be.index]
    return be

def get_es1_gold():
    vad = pd.read_excel(cs.es1)
    rename = {'Val-Mn-All':'valence',
              'Aro-Mn-All':'arousal',
              'Dom-Mn-All':'dominance',
              'S-Word': 'word'}
    vad.rename(columns=rename,
                   inplace=True)
    vad.set_index('word', inplace=True)
    for c in vad.columns:
        if c not in rename.values():
            vad.drop(columns=c, inplace=True)

    return vad


def get_es2_gold():
    stadthagen16 = pd.read_csv(cs.es2, encoding = 'cp1252')
    stadthagen16 = stadthagen16[['Word', 'ValenceMean', 'ArousalMean']]
    stadthagen16.columns = ['word', 'valence', 'arousal']
    stadthagen16.set_index('word', inplace = True)
    return stadthagen16


def get_es3_gold():
    hinojosa16a = pd.read_excel(cs.es3)
    hinojosa16a = hinojosa16a[['Word','Val_Mn', 'Ar_Mn']]
    hinojosa16a.columns = ['word', 'valence', 'arousal']
    hinojosa16a.set_index('word', inplace = True)
    return hinojosa16a


def get_es4_gold():
    hinojosa16a = pd.read_excel(cs.es4)
    hinojosa16a = hinojosa16a[['Word', 'Hap_Mn', 'Ang_Mn','Sad_Mn',
                             'Fear_Mn', 'Disg_Mn']]
    hinojosa16a.columns = ['word', 'joy','anger','sadness','fear','disgust']
    hinojosa16a.set_index('word', inplace = True)
    return hinojosa16a


def get_es5_gold():
    df = pd.read_csv(cs.es5, encoding = 'cp1252')
    df = df[['Word',  'Happiness_Mean', 'Anger_Mean', 'Sadness_Mean', 'Fear_Mean', 'Disgust_Mean']]
    df.columns = ['word','joy','anger', 'sadness', 'fear','disgust']
    df.set_index('word', inplace = True)
    return df


def get_es6_gold():
    be = pd.read_excel(cs.es6)
    rename = {
        'Hap_Mean': 'joy',
        'Ang_Mean': 'anger',
        'Sad_Mean': 'sadness',
        'Fear_Mean': 'fear',
        'Disg_Mean': 'disgust',
        'Spanish_Word': 'word',
    }

    be = be.rename(columns=rename)
    be.set_index('word', inplace=True)
    for c in be.columns:
        if c not in rename.values():
            be.drop(columns=c, inplace=True)
    return be


def get_de1_gold():
    de_gold = pd.read_excel(cs.de1)
    rename ={'VAL_Mean':'valence',
             'ARO_Mean_(ANEW)':'arousal',
              'DOM_Mean': 'dominance',
             'G-word': 'word'}

    de_gold.rename(columns=rename, inplace=True)
    for c in de_gold.columns:
        if c not in rename.values():
            de_gold.drop(columns=c, inplace=True)

    # make lower case to match embeddings
    #de_gold.target = de_gold.target.apply(func = (lambda x: x.lower()))
    de_gold.set_index('word', inplace=True)
    de_gold = de_gold[~de_gold.index.duplicated()]
    return de_gold


def get_de2_gold():
    df = pd.read_excel(cs.de2, index_col=1)
    df.index.rename('word', inplace = True)

    dct = {
        'EMO_MEAN': 'valence',
        'AROUSAL_MEAN': 'arousal',
        'HAP_MEAN': 'joy',
        'ANG_MEAN': 'anger',
        'SAD_MEAN': 'sadness',
        'FEA_MEAN': 'fear',
        'DIS_MEAN': 'disgust'
    }

    df = df.rename(columns=dct)
    for c in df.columns:
        if c not in dct.values():
            df.drop(columns=c, inplace=True)

    return df


def get_de3_gold():
    with open(cs.de3, encoding = 'cp1252') as f:
        kanske10 = f.readlines()
    # Filtering out the relevant portion of the provided file
    kanske10 = kanske10[7: 1008]
    #kanske10 = kanske10[7:]
    # Creating data frame from string:
    #https: //stackoverflow.com/questions/22604564/how-to-create-a-pandas-dataframe-from-string
    kanske10 = pd.read_csv(StringIO(''.join(kanske10)), sep = '\t')
    kanske10 = kanske10[['word', 'valence_mean','arousal_mean']]
    kanske10.columns = ['word', 'valence', 'arousal']
    kanske10['word'] = kanske10['word'].str.lower()
    kanske10.set_index('word', inplace = True)
    kanske10 = kanske10[~kanske10.index.duplicated()]
    #kanske10 = kanske10.asd
    return kanske10


def get_de4_gold():
    vad = pd.read_excel(cs.de2, index_col=1)
    vad.index.rename('word', inplace = True)

    be = pd.read_excel(cs.de4, index_col=1)
    be.index.rename('word', inplace = True)

    df = vad.join(be, how='inner', rsuffix='r')
    new_index = []
    for i in range(len(df)):
        w = df.index[i]
        pos = df.iloc[i]['WORD_CLASS']

        if pos == 'N':
            w = w[0].upper() + w[1:]
        new_index.append(w)
    df.index = new_index

    dct = {
        'HAP_MEAN': 'joy',
        'ANG_MEAN': 'anger',
        'SAD_MEAN': 'sadness',
        'FEA_MEAN': 'fear',
        'DIS_MEAN': 'disgust'
    }

    df = df.rename(columns=dct)
    for c in df.columns:
        if c not in dct.values():
            df.drop(columns=c, inplace=True)

    return df


def get_pl1_gold():
    pl_gold = pd.read_excel(cs.pl1, index_col=0)
    rename = {'Valence_M':'valence',
              'arousal_M':'arousal',
              'dominance_M':'dominance',
              'polish word': 'word',
             }
    pl_gold.rename(columns=rename,
                   inplace=True)
    pl_gold.set_index('word', inplace=True)

    for c in pl_gold.columns:
        if c not in rename.values():
            pl_gold.drop(columns=c, inplace=True)

    return pl_gold


def get_pl2_gold():
    vad = pd.read_excel(cs.pl2, index_col=2)
    vad.index.rename('word', inplace=True)

    be = pd.read_excel(cs.pl3, index_col=2)
    be.index.rename('word', inplace=True)

    df = vad.join(be, how='inner', rsuffix='r')
    dc = {
        'val_M_all': 'valence',
        'aro_M_all': 'arousal'
    }
    df.rename(columns=dc, inplace=True)
    for c in df.columns:
        if c not in dc.values():
            df.drop(columns=c, inplace=True)

    return df


def get_pl3_gold():
    vad = pd.read_excel(cs.pl2, index_col=2)
    vad.index.rename('word', inplace=True)

    be = pd.read_excel(cs.pl3, index_col=2)
    be.index.rename('word', inplace=True)

    df = vad.join(be, how='inner', rsuffix='r')
    dc = {
        'hap_M_all': 'joy',
        'ang_M_all': 'anger',
        'sad_M_all': 'sadness',
        'fea_M_all': 'fear',
        'dis_M_all': 'disgust'
    }
    df.rename(columns=dc, inplace=True)
    for c in df.columns:
        if c not in dc.values():
            df.drop(columns=c, inplace=True)

    return df


def get_zh1_gold():
    '''
    Yu, L.-C., Lee, L.-H., Hao, S., Wang, J., He, Y., Hu, J., … Zhang, X.
    (2016). Building Chinese Affective Resources in Valence-Arousal Dimensions.
    In Proceedings of NAACL-2016.
    '''
    #path = RESOURCES / 'Yu-2016-Chinese Valence Arousal Words (CVAW)/v2/cvaw2.csv'
    #path = TARGETGOLD / 'cvaw2_simplified.csv'
    yu16 = pd.read_csv(cs.zh1)
    yu16 = yu16[['Word_simplified', 'Valence_Mean', 'Arousal_Mean']]
    yu16.columns=['Word','valence', 'arousal']
    yu16.set_index('Word', inplace=True)
    yu16 = yu16[~yu16.index.duplicated()]

    return yu16


def get_zh2_gold():
    '''
    Yao, Z., Wu, J., Zhang, Y., & Wang, Z. (2016). Norms of valence, arousal,
    concreteness, familiarity, imageability, and context availability for
    1,100 Chinese words. Behavior Research Methods.
    '''
    zh_gold = pd.read_excel(cs.zh2, index_col=0)
    zh_gold.index.rename('word', inplace=True)
    rename_dict = {'VAL_M': 'valence', 'ARO_M': 'arousal'}
    zh_gold = zh_gold[list(rename_dict.keys())]
    zh_gold.rename(columns=rename_dict, inplace=True)
    return zh_gold

def get_it_gold():
    montefinese14 = pd.read_excel(cs.it, header=1)
    montefinese14 = montefinese14[['Ita_Word', 'Eng_Word', 'M_Val', 'M_Aro', 'M_Dom']]
    montefinese14 = montefinese14.rename(columns={'M_Val': 'valence',
                                                  'M_Aro': 'arousal',
                                                  'M_Dom': 'dominance',
                                                  'Ita_Word': 'word',
                                                  'Eng_Word': 'source'})

    gold_it = montefinese14.set_index('word').drop(columns='source')
    return gold_it

def get_pt_gold():
    pt_gold = pd.read_excel(cs.pt, sheet_name=1)

    rename = {
        'EP-Word': 'word',
        'Val-M': 'valence',
        'Arou-M': 'arousal',
        'Dom-M': 'dominance'
    }

    pt_gold.rename(columns=rename, inplace=True)

    for c in pt_gold.columns:
        if c not in rename.values():
            pt_gold.drop(columns=c, inplace=True)

    return pt_gold.set_index('word')

def get_nl_gold():
    nl_gold = pd.read_excel(cs.nl, header=1)

    rename = {
        'Words': 'word',
        'M V': 'valence',
        'M A': 'arousal',
    }

    nl_gold.rename(columns=rename, inplace=True)

    for c in nl_gold.columns:
        if c not in rename.values():
            nl_gold.drop(columns=c, inplace=True)
    return nl_gold.set_index('word')

def get_id_gold():
    id_gold = pd.read_excel(cs.id)

    rename = {
        'Words (Indonesian)': 'word',
        'ALL_Valence_Mean': 'valence',
        'ALL_Arousal_Mean': 'arousal',
        'ALL_Dominance_Mean': 'dominance'
    }

    id_gold.rename(columns=rename, inplace=True)
    for c in id_gold.columns:
        if c not in rename.values():
            id_gold.drop(columns=c, inplace=True)

    id_gold.set_index('word', inplace=True)
    id_gold = id_gold[~id_gold.index.duplicated()]
    return id_gold

def get_el_gold():
    df = pd.read_csv(cs.el, index_col=1)
    df.index = df.index.rename('word')
    dc = {'Valence': 'valence',
          'Arousal': 'arousal',
          'Dominance': 'dominance'}
    df = df.rename(columns=dc)
    for c in df.columns:
        if c not in dc.values():
            df.drop(columns=c, inplace=True)

    return df

def get_tr1_gold():
    df = pd.read_csv(cs.tr, sep=';', index_col=0)
    dc = {
        'ValenceM': 'valence',
        'ArousalM': 'arousal',
    }
    df = df.rename(columns=dc)
    for c in df.columns:
        if c not in dc.values():
            df = df.drop(columns=c)

    df.index = [w.lower() for w in df.index]
    df = df[~df.index.duplicated()]

    return df

def get_tr2_gold():
    df = pd.read_csv(cs.tr, sep=';', index_col=0)
    dc = {
        'HappyM': 'joy',
        'AngerM': 'anger',
        'SadM': 'sadness',
        'FearM': 'fear',
        'DisgustM': 'disgust',
    }
    df = df.rename(columns=dc)
    for c in df.columns:
        if c not in dc.values():
            df = df.drop(columns=c)

    df.index = [w.lower() for w in df.index]
    df = df[~df.index.duplicated()]

    return df

def get_hr_gold():
    df = pd.read_excel(cs.hr, index_col=1)
    df.index.rename('word', inplace=True)
    dc = {
        'Val_Total_Mean': 'valence',
        'Aro_Total_Mean': 'arousal',
    }
    df = df.rename(columns=dc)
    for c in df.columns:
        if c not in dc.values():
            df = df.drop(columns=c)


    return df

conditions = {
    'en1': {'iso': 'en',
            'emo': 'vad',
            'citation': 'xanew',
            'get': get_en1_gold},
    'en2': {'iso': 'en',
            'emo': 'vad',
            'citation': 'bradley99',
            'get': get_en2_gold},
    'en3': {'iso': 'en',
            'emo': 'be',
            'citation': 'stevenson07',
            'get': get_en3_gold},
    'es1': {'iso': 'es',
            'emo': 'vad',
            'citation': 'redondo07',
            'get': get_es1_gold},
    'es2': {'iso': 'es',
            'emo': 'vad',
            'citation': 'Stadthagen16',
            'get': get_es2_gold},
    'es3': {'iso': 'es',
            'emo': 'vad',
            'citation': 'hiojosa16',
            'get': get_es3_gold},
    'es4': {'iso': 'es',
            'emo': 'be',
            'get': get_es4_gold,
            'citation': 'hinojosa 16'},
    'es5': {'iso': 'es',
            'emo': 'be',
            'citation': 'Stadthagen17',
            'get': get_es5_gold},
    'es6': {'iso': 'es',
            'emo': 'be',
            'citation': 'ferre',
            'get': get_es6_gold},
    'de1': {'iso': 'de',
            'emo': 'vad',
            'get': get_de1_gold,
            'citation': 'Schmidtke14'},
    'de2': {'iso': 'de',
            'emo': 'vad',
            'citation': 'Vo09',
            'get': get_de2_gold},
    'de3': {'iso': 'de',
            'emo': 'vad',
            'citation': 'Kanske10',
            'get': get_de3_gold},
    'de4': {'iso': 'de',
            'emo': 'be',
            'citation': 'Briesemeister12',
            'get': get_de4_gold},
    'pl1': {'iso': 'pl',
            'emo': 'vad',
            'citation': 'Imbir',
            'get': get_pl1_gold},
    'pl2': {'iso': 'pl',
            'emo': 'vad',
            'citation':'Riegel15',
            'get': get_pl2_gold},
    'pl3': {'iso': 'pl',
            'emo': 'be',
            'citation': 'wierzba',
            'get': get_pl3_gold},
    'zh1': {'iso': 'zh',
           'emo': 'vad',
           'citation': 'yu16',
           'get': get_zh1_gold},
    'zh2': {'iso': 'zh',
           'emo': 'vad',
           'citation': 'yao16',
           'get': get_zh2_gold},
    'it': {'iso': 'it',
           'emo': 'vad',
           'citation': 'montefinese',
           'get': get_it_gold},
    'pt': {'iso': 'pt',
           'emo': 'vad',
           'citation': 'soares12',
           'get': get_pt_gold},
    'nl': {'iso': 'nl',
           'emo': 'vad',
           'citation': 'Moors13',
           'get': get_nl_gold},
    'id': {'iso': 'id',
           'emo': 'vad',
           'citation': 'sianipar',
           'get': get_id_gold},
    'el': {'iso': 'el',
           'emo': 'vad',
           'citation': 'palogiannidi',
           'get': get_el_gold},
    'tr1': {'iso': 'tr',
            'emo': 'vad',
            'citation': 'Kapucu',
            'get': get_tr1_gold},
    'tr2': {'iso': 'tr',
            'emo': 'be',
            'citation': 'Kapucu',
            'get': get_tr2_gold},
    'hr': {'iso': 'hr',
           'emo': 'vad',
           'citation': 'Coso',
           'get': get_hr_gold}
}

vad_lexica = [key for key, value in conditions.items() if value['emo']=='vad' or value['emo']=='va']
be_lexica = [key for key, value in conditions.items() if value['emo']=='be']



def formatter(x):
    if np.isnan(x):
        return '---'
    else:
        return "{:.2f}".format(x).lstrip('0')


