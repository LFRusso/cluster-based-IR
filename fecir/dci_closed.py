# função para obter a entrada do dci_closed = CLOSED_SET, PRE_SET, POST_SET
def extract_itemsets(cluster, min_supp):
    pre_set = []

    # pegar itens que estão em todos os documentos = c(∅)
    closed_set = []
    for item in pre_set:
        ver = 0
        for doc in cluster:
            if item in doc:
                ver += 1

        if ver == len(cluster):
            closed_set.append(item)

    # seleção dos termos frequentes
    f1 = []
    _ = [f1.append(item) for doc in cluster for item in doc if supp([item], cluster) >= min_supp and item not in f1]

    # itens frequentes que não estão no closed = F1 \ c(∅)
    post_set = [item for item in f1 if item not in closed_set]

    return closed_set, pre_set, post_set


def dci_closed(closed_set, pre_set, post_set, cluster, min_supp):
    subsets = []
    for i in range(len(post_set)):
        new_gen = closed_set.copy()
        new_gen.append(post_set[i])
        if supp(new_gen, cluster) >= min_supp:
            if not is_dup(new_gen, pre_set, cluster):
                closed_set_new = new_gen.copy()
                post_set_new = []
                for j in range(i+1, len(post_set)):
                    if set(g(new_gen, cluster)).issubset(g(post_set[j], cluster)):
                        closed_set.append(post_set[j])
                    else:
                        post_set_new.append(post_set[j])

                subsets.append(closed_set_new)
                retorno = dci_closed(closed_set_new, pre_set, post_set_new, cluster, min_supp)
                subsets.extend(retorno)
                pre_set.append(post_set[i])

    return subsets


# cálculo do support
def supp(itemset, cluster):
    support = 0
    for doc in cluster:
        if set(itemset).issubset(doc):
            support += 1
    return support/len(cluster)


# checagem de duplicados
def is_dup(new_gen, pre_set, cluster):
    for j in pre_set:
        if set(g(new_gen, cluster)).issubset(g(j, cluster)):
            return True

    return False


def g(itemset, cluster):
    lista_de_trans = []
    for doc in range(len(cluster)):
        if set(itemset).issubset(cluster[doc]):
            lista_de_trans.append(doc)

    return lista_de_trans
