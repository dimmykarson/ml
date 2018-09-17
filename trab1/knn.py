
import csv, operator, gensim, sys, random, time, math, gc
from random import shuffle
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import cosine, euclidean, cityblock, mahalanobis
from functools import wraps

import warnings
warnings.filterwarnings("ignore")
np.warnings.filterwarnings("ignore")


len_leaf = 1000
len_train = 25000
PROF_DATA = {}



def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time
        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)
        return ret
    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print "Function %s called %d times. " % (fname, data[0]),
        print 'Execution time max: %.3f, average: %.3f' % (max_time, avg_time)
def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}

class Tree:
    def __init__(self, data=None, parent=None, left=None, right=None, i_s=None, lenght=None):
        self.parent = parent
        self.left = left
        self.right = right
        self.data = None
        if not lenght:
            lenght = len(data[0][4])
        if not i_s:
            i_s = range(0, lenght)
        self.mount(data, i_s)

    def mount(self, data, i_s=None):
        if len(data)<len_leaf:
            self.data = data
            return
        i = random.choice(i_s)
        z = [x[4][i] for x in data]
        serie = pd.Series(z)
        mediana = serie.median()
        self.mediane = mediana
        self.index = i
        i_s.remove(i)
        a = [x for x in data if x[4][i] <= mediana]
        b = [x for x in data if x[4][i] > mediana]

        self.left = Tree(data=a, parent=self, i_s=i_s[:])
        self.right = Tree(data=b, parent=self, i_s=i_s[:])

@profile
def load_data(filename, length=-1):
    data = []
    sentences = []
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            z = [x.replace(",", "").replace(".", "").replace("\"", "") for x in row[2].split() if len(x)>2]
            sentences.append(z)
            data.append([row[0], row[1], row[2], row[3], z])
            if length>0 and len(data) > length:
                break
        csvfile.close()
    random.shuffle(sentences)
    model = gensim.models.Word2Vec(sentences, min_count=1, size=300)
    for d in data:
        target = []
        for p in d[4]:
            v = model[p]
            target.append(v)
        median_vector = np.median(target, axis=0)
        np.median(target, axis=0, out=median_vector)
        d[4] = median_vector

    #normalize
    n = [x[4] for x in data]
    n = np.array(n)
    n_normed = n / n.max()
    for i in range(len(n_normed)):
        data[i][4] = n_normed[i]
    return data


@profile
def get_tree(data):
    return Tree(data)


@profile
def calculo_similaridade(a, b, type="euclidean"):
    if len(a)!=len(b):
        raise Exception("Representacoes com tamanhos diferentes")
    if type == 'euclidean':
        return distancia_euclideana(a, b)
    elif type == 'manhattan':
        return distancia_manhattan(a, b)
    elif type == 'cosine':
        return distancia_cosseno(a, b)
    elif type == 'mahalanobis':
        return distancia_mahalanobis(a, b)


def distancia_euclideana(a, b):
	return euclidean(a, b)

def distancia_manhattan(a, b):
    return cityblock(a, b)

def distancia_cosseno(a, b):
    return cosine(a, b)

def distancia_mahalanobis(a, b):
    d = {}
    d["a"] = a
    d["b"] = b
    df = pd.DataFrame(d)
    covmx = df.cov()
    invcovmx = sp.linalg.inv(covmx)
    return mahalanobis(a, b, invcovmx)

@profile
def vizinhos(tree, test, k, type ="euclidean"):
    distancias = []
    train = get_conjunto(tree, test)
    for x in range(len(train)):
		dist = calculo_similaridade(test[4], train[x][4], type)
		distancias.append((train[x], dist))
    distancias.sort(key=lambda x: x[1])
    return [x[0] for x in distancias[:k]]

@profile
def votacao(vizinhos):
    classes_votadas = {}
    classes_votadas["pos"] = 0
    classes_votadas["neg"] = 0
    for vizinho in vizinhos:
        classe = vizinho[3]
        if classe=="pos":
            classes_votadas["pos"] += 1
        else:
            classes_votadas["neg"] += 1
    votacao = sorted(classes_votadas.iteritems(), key=operator.itemgetter(1), reverse=True)
    return votacao[0][0]

@profile
def precisao(test, predicoes):
    correto = 0
    mconfusao = {}
    mconfusao["VP"] = 0
    mconfusao["VN"] = 0
    mconfusao["FP"] = 0
    mconfusao["FN"] = 0
    for i in range(len(test)):
        if test[i][3]=="pos" and predicoes[i] == "pos":
            mconfusao["VP"] +=1
        if test[i][3]=="neg" and predicoes[i] == "neg":
            mconfusao["VN"] += 1
        if test[i][3] == "neg" and predicoes[i] == "pos":
            mconfusao["FP"] += 1
        if test[i][3] == "pos" and predicoes[i] == "neg":
            mconfusao["FN"] += 1
    n = mconfusao["VP"] + mconfusao["FP"] + mconfusao["VN"] + mconfusao["FN"]
    taxa_erro_positiva = 0
    taxa_erro_negativa = 0
    taxa_erro_total = 0
    confiabilidade_positiva = 0
    confiabilidade_negativa = 0
    try:
        confiabilidade_positiva = float(mconfusao["VP"])/(float(mconfusao["VP"])+float(mconfusao["FP"]))
        confiabilidade_negativa = float(mconfusao["VN"]) / (float(mconfusao["VN"]) + float(mconfusao["FN"]))
        taxa_erro_positiva = float(mconfusao["FN"])/(float(mconfusao["VP"])+float(mconfusao["FN"]))
        taxa_erro_negativa = float(mconfusao["FP"]) / (float(mconfusao["FP"]) + float(mconfusao["VN"]))
        taxa_erro_total = (float(mconfusao["FP"])+float(mconfusao["FN"]))/n
    except:
        pass
    acuracia = (float(mconfusao["VP"])+float(mconfusao["VN"]))/float(n)
    return acuracia, mconfusao, taxa_erro_positiva, taxa_erro_negativa, taxa_erro_total, confiabilidade_positiva, confiabilidade_negativa

def print_matrix(matrix):
    print "\t|\tP\t\t|\tN\t\t|\n" \
          "P\t|\t{0}\t|\t{1}\t|\n" \
          "N\t|\t{2}\t|\t{3}\t|\n".format(matrix["VP"], matrix["FN"], matrix["FP"], matrix["VN"])

@profile
def get_conjunto(tree, test):
    if tree.data:
        return tree.data
    else:
        v = test[4][tree.index]
        if v <= tree.mediane:
            return get_conjunto(tree.left, test)
        else:
            return get_conjunto(tree.right, test)
    return None

@profile
def main(train, test, k, distance_type="euclidean"):
    print "Carregando dados de treinamento"
    train = load_data(train, length=len_train)
    print "Tamanho do treinamento: {0}".format(len(train))
    print "Montando arvore"
    root = get_tree(train)
    print "Carregando dados de Teste"
    test = load_data(test, length=len_train)
    print "Tamanho do teste: {0}".format(len(test))
    print "Treinamento... k={0}, distancia: {1}".format(k, distance_type)
    preds = []
    i = 0
    for t in test:
        i += 1
        nn = vizinhos(root, t, k, distance_type)
        resultado_votacao = votacao(nn)
        preds.append(resultado_votacao)
    scores = precisao(test, preds)
    print "Precisao: {0}%".format(scores[0])
    print "Matrix"
    print_matrix(scores[1])

def run():
    main("train.csv", "validation.csv", 20, distance_type="cosine")
    print_prof_data()

def teste():
    ks = [3, 5, 10, 20, 30]
    distancias = [
        # "euclidean",
        "manhattan",
        "cosine",
    ]
    print "Carregando dados de teste"
    train = load_data("train.csv", length=len_train)
    root = get_tree(train)
    test = load_data("validation.csv", length=len_train)
    resultado = open("resultado.txt", "w")
    print "Vamos comecar"
    for dist in distancias:
        for k in ks:
            print "Treinamento... k={0}, distancia: {1}".format(k, dist)
            preds = []
            i = 0
            for t in test:
                i += 1
                nn = vizinhos(root, t, k, dist)
                resultado_votacao = votacao(nn)
                preds.append(resultado_votacao)
            scores = precisao(test, preds)
            resultado.write("====== k = {0}, distancia: {1}\n".format(k, dist))
            resultado.write("Precisao: {0}\n".format(scores[0]))
            resultado.write("\t|\tP\t\t|\tN\t\t|\n" \
                            "P\t|\t{0}\t|\t{1}\t|\n" \
                            "N\t|\t{2}\t|\t{3}\t|\n".format(scores[1]["VP"], scores[1]["FN"], scores[1]["FP"], scores[1]["VN"]))
            resultado.write("\n\n")
            print "Precisao: {0}%".format(scores[0])
            print "Matrix"
            print_matrix(scores[1])




if __name__ == "__main__":
        if len(sys.argv) != 4:
                sys.exit("Erro nos parametros")
        main(sys.argv[1], sys.argv[2], sys.argv[3])