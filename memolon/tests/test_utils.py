import pandas as pd
from memolon.src import utils

def test_get_source():

    train = utils.get_source('train')
    test = utils.get_source('test')
    dev = utils.get_source('dev')

    for split in [train, dev, test]:
        assert list(split.columns) == ["valence", "arousal", "dominance", "joy", "anger", "sadness", "fear", "disgust"]

        assert len(train) == 11463
        assert len(dev) == 1296
        assert len(test) == 1032



def test_get_TargetPred():
    for iso in utils.language_table.index:
        assert isinstance(utils.get_TargetPred(iso), pd.DataFrame)



def test_get_TargetGold():
    for key, cond in utils.conditions.items():
        df = cond["get"]()
        assert isinstance(df, pd.DataFrame)

        # check column names for correctness
        actual = list(df.columns)
        if cond["emo"] == "vad":
            assert actual == ["valence", "arousal", "dominance"] or actual ==["valence", "arousal"]
        else:
            assert actual == ["joy", "anger", "sadness", "fear", "disgust"]