"""
preprocessing.py

Defines common preprocessing steps for text classification pipelines using PySpark.
"""

from preprocessing.text_preprocessing import TextCleaner, Stemmer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

def get_preprocessing_stages(use_idf=True):
    """
    Creates a list of preprocessing stages for Spark ML pipelines.

    Args:
        use_idf (bool): Whether to include the IDF stage after HashingTF. If False,
                        HashingTF will output directly to 'features'.

    Returns:
        List[pyspark.ml.PipelineStage]: Ordered preprocessing stages.
    """
    text_cleaner = TextCleaner(inputCol="text", outputCol="cleaned_text")
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    stopwords = StopWordsRemover(inputCol="words", outputCol="filtered")
    stemmer = Stemmer(inputCol="filtered", outputCol="stemmed")
    hashing_tf = HashingTF(inputCol="stemmed", outputCol="raw_features")

    if use_idf:
        idf = IDF(inputCol="raw_features", outputCol="features")
        return [text_cleaner, tokenizer, stopwords, stemmer, hashing_tf, idf]
    else:
        hashing_tf.setOutputCol("features")
        return [text_cleaner, tokenizer, stopwords, stemmer, hashing_tf]

def build_preprocessing_pipeline(use_idf=True):
    """
    Returns a Spark Pipeline object that performs text preprocessing.
    
    Args:
        use_idf (bool): Whether to include IDF stage after HashingTF.
        
    Returns:
        Pipeline: A Spark ML pipeline with text preprocessing stages.
    """
    stages = get_preprocessing_stages(use_idf=use_idf)
    return Pipeline(stages=stages)
