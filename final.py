import random
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics.pairwise import cosine_distances as cd
from scipy.cluster.hierarchy import dendrogram

def save_dendrogram(M, labels, name):
    n = M.shape[0]
    clusters = np.arange(n)
    sizes = {i: 1 for i in range(n)}
    Z = []
    next_ = n
    levels = sorted(set(M[i, j] for i in range(n) for j in range(i + 1, n)))
    for level in levels:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if M[i, j] == level]
        for i, j in pairs:
            c_i = clusters[i]
            c_j = clusters[j]
            if c_i == c_j:
                continue
            size = sizes[c_i] + sizes[c_j]
            Z.append([c_i, c_j, float(level), size])
            sizes[next_] = size
            for k in range(n):
                if clusters[k] in (c_i, c_j):
                    clusters[k] = next_
            next_ += 1

    Z = np.array(Z)

    plt.figure(figsize=(10, 6))
    data = dendrogram(Z, labels = labels, leaf_rotation = 90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Merge Level")
    plt.tight_layout()
    plt.savefig(name, dpi = 100)

def run_flow(name, levels = 10, categories = ['comp.sys.mac.hardware', 'rec.sport.baseball']):
    data = fetch_20newsgroups(categories = categories, random_state = random.randint(1, 1000), shuffle = True, remove = ("headers", "footers", "quotes"))
    start, end = 370, 395
    docs = data.data[start: end]
    vectorizer = TfidfVectorizer(max_df = 0.3, stop_words = 'english')
    vectors = vectorizer.fit_transform(docs)
    L = levels
    num_docs = vectors.shape[0]
    labels = data.target[start: end]
    doc_labels = [data.target_names[label] for label in labels]
    node_triples = []

    D = cd(vectors)
    for a in range(num_docs):
        for b in range(num_docs):
            if b == a:
                continue
            for c in range(b + 1, num_docs):
                if c == a or c == b:
                    continue
                node_triples.append((a, b, c))
    w = {}
    for a, b, c in node_triples:
        w[(a, b, c)] = float(D[a, c] - D[a, b])
        
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    M = model.addVars(num_docs, num_docs, vtype = GRB.INTEGER, lb = 1, ub = L, name = "M")
    O = model.addVars(node_triples, vtype = GRB.BINARY, name = "O")
    Z_ab_ac = model.addVars(node_triples, vtype = GRB.BINARY, name = "Z_ab_ac")
    Z_bc_ac = model.addVars(node_triples, vtype = GRB.BINARY, name = "Z_bc_ac")
    
    for a in range(num_docs):
        model.addConstr(M[a, a] == 1)
        
    for a in range(num_docs):
        for b in range(num_docs):
            model.addConstr(M[a, b] ==  M[b, a])

    for (a, b, c) in node_triples:
        model.addConstr(M[a, c] - M[a, b] - (L + 1) * O[(a, b, c)] >= -L)
        model.addConstr(M[a, b] - M[a, c] <= (1 - O[(a, b, c)]) * L)
        model.addConstr(M[a, b] - M[a, c] - (L + 1) * Z_ab_ac[(a, b, c)] + 1 >= -L)
        model.addConstr(M[a, b] - M[a, c] - (L + 1) * Z_ab_ac[(a, b, c)] + 1 <= 0)
        model.addConstr(M[b, c] - M[a, c] - (L + 1) * Z_bc_ac[(a, b, c)] + 1 >= -L)
        model.addConstr(M[b, c] - M[a, c] - (L + 1) * Z_bc_ac[(a, b, c)] + 1 <= 0)
        model.addConstr(Z_ab_ac[(a, b, c)] + Z_bc_ac[(a, b, c)] >= 1)
    
    model.setObjective(gp.quicksum(w[(a, b, c)] * O[(a, b, c)] for (a, b, c) in node_triples), GRB.MAXIMIZE)
    model.Params.MIPGap = 0.03
    model.Params.TimeLimit = 180
    model.optimize()

    M_opt = np.zeros((num_docs, num_docs), dtype=int)

    for (i, j), var in M.items():
        val = int(var.X)
        M_opt[i, j] = val
        M_opt[j, i] = val

    save_dendrogram(M_opt, doc_labels, name)

run_flow(name = '2_categories.png')
run_flow(name = '3_categories.png', categories = ['comp.sys.mac.hardware', 'rec.sport.baseball', 'talk.religion.misc'])
run_flow(name = '4_categories.png', categories = ['comp.sys.mac.hardware', 'rec.sport.baseball', 'talk.politics.mideast', 'rec.sport.hockey'])
