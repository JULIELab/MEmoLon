from pathlib import Path

vad = ['valence', 'arousal', 'dominance']
be5 = ['joy', 'anger', 'sadness', 'fear', 'disgust']
emotions = vad + be5


PROJECT_ROOT = Path(__file__).parents[2].resolve()
DATA = PROJECT_ROOT / "memolon" / "data"
ANALYSES = PROJECT_ROOT / "memolon" / "analyses"

SOURCE = DATA / "Source"
TARGETGOLD = DATA / "TargetGold"
TARGETPRED = DATA / "TargetPred"
TARGETPRED_MTL_GROUPED = TARGETPRED / "MTL_grouped"
TARGETPRED_RIDGE = TARGETPRED / "ridge"
TARGETPRED_STL = TARGETPRED / "STL"
TARGETPRED_MTL_all = TARGETPRED / "MTL_all"


EMBEDDINGS = DATA / "Embeddings"
TRANSLATION = DATA / "TranslationTables"

language_table = DATA / "languages_overview.tsv"

# Source lexica
warriner_vad = SOURCE / "Ratings_Warriner_et_al.csv"
warriner_be = SOURCE / "Warriner_BE.tsv"

# Gold lexica
en2 = TARGETGOLD / "ANEW1999.tsv"
en3 = TARGETGOLD / "Stevenson(2007)-ANEW_emotional_categories.xls"
es1 = TARGETGOLD / "Redondo(2007).xls"
es2 = TARGETGOLD / "Stadthagen_VA.csv"
es3 = TARGETGOLD / "Hinojosa et al_Supplementary materials.xlsx"
es4 = TARGETGOLD / "Hinojosa et al_Supplementary materials.xlsx"
es5 = TARGETGOLD / "Stadthagen_BE.csv"
es6 = TARGETGOLD / "Ferre.xls"
de1 = TARGETGOLD / "Schmidtke.xlsx"
de2 = TARGETGOLD / "BAWL-R.xls"
de3 = TARGETGOLD / "LANG_database.txt"
de4 = TARGETGOLD / "Briesemeister.xls"
pl1 = TARGETGOLD / "Imbir.xlsx"
pl2 = TARGETGOLD / "Riegel.xlsx"
pl3 = TARGETGOLD / "Wierzba.xlsx"
zh1 = TARGETGOLD / "cvaw2_simplified.csv"
zh2 = TARGETGOLD / "Yao.xlsx"
it = TARGETGOLD / "Montefinese.xls"
pt = TARGETGOLD / "Soares.xls"
nl = TARGETGOLD / "Moors.xlsx"
id = TARGETGOLD / "Sianipar.xlsx"
el = TARGETGOLD / "greek_affective_lexicon.csv"
tr = TARGETGOLD / "TurkishEmotionalWordNorms.csv"
hr = TARGETGOLD / "Coso.xlsx"

GOLD_EVALUATION_RESULTS = ANALYSES / "gold_evaluation.csv"
GOLD_LEXICA_OVERVIEW = ANALYSES / "gold_lexica.csv"
COMPARISON_AGAINST_HUMAN_RELIABILITY = ANALYSES / "comparison_against_human_reliability.json"
TRANSLATION_VS_PREDICTION_RESULTS = ANALYSES / "translation_vs_prediction.csv"

SILVER_EVALUATION_RESULTS = ANALYSES / "silver_evaluation.csv"
BASELINE_RESULTS = ANALYSES / "baseline_results.csv"
DEV_EXPERIMENT_RESULTS = ANALYSES / "dev_experiment_results.csv"
GOLD_SILVER_AGREEMENT = ANALYSES / "gold_silver_agreement.csv"
GENERATED_LEXICA_OVERVIEW = ANALYSES / "generated_lexica.csv"









