def func1():
    import thulac
    thu1 = thulac.thulac()
    text = thu1.cut("I love you", text=True)
    print(text)
def func2():
    import gensim
    from gensim.models import Word2Vec
    from gensim.test.utils import common_texts
    docs = [['here','parameter'],['window','play']]
    # print(common_texts)
    # model = Word2Vec(sentences=docs, size=100, window=5, min_count=1, workers=4)
    model = gensim.models.KeyedVectors.load_word2vec_format('data/vectors.txt', binary=False)
    print(model.vocab.keys())


    # model = gensim.models.Word2Vec(sentences=common_texts,
    #                           min_count=10, 
    #                           workers=4,
    #                           size=50,
    #                           window=5,
    #                           iter = 10)
if __name__ == "__main__":
    # func1()
    func2()