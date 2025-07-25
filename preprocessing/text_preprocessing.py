import uuid
import string
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from nltk.stem.snowball import SnowballStemmer

class TextCleaner(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="text", outputCol="cleaned_text"):
        super(TextCleaner, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.uid = "TextCleaner_" + str(uuid.uuid4())

    def _transform(self, df):
        def clean_text(text):
            if text:
                text = text.lower()
                return text.translate(str.maketrans('', '', string.punctuation))
            return ""
        clean_udf = udf(clean_text, StringType())
        return df.withColumn(self.outputCol, clean_udf(df[self.inputCol]))

    def transformSchema(self, schema):
        return schema.add(self.outputCol, StringType(), nullable=True)

    def copy(self, extra=None):
        return TextCleaner(self.inputCol, self.outputCol)


class Stemmer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="filtered", outputCol="stemmed"):
        super(Stemmer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.uid = "Stemmer_" + str(uuid.uuid4())

    def _transform(self, df):
        def stem_words(words):
            stemmer = SnowballStemmer("english")
            return [stemmer.stem(word) for word in words if word is not None]
        stem_udf = udf(stem_words, ArrayType(StringType()))
        return df.withColumn(self.outputCol, stem_udf(df[self.inputCol]))

    def transformSchema(self, schema):
        return schema.add(self.outputCol, ArrayType(StringType()), nullable=True)

    def copy(self, extra=None):
        return Stemmer(self.inputCol, self.outputCol)