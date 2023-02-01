import numpy as np
# np.random.seed(4)
import tensorflow.keras.backend as K
from collections import defaultdict


def select_from_large(select_amount, target_lsa):
    selected_lst, lsa_lst = order_output(target_lsa, select_amount)
    # print(lsa_lst)
    print(selected_lst)
    selected_index = []
    for i in range(select_amount):
        selected_index.append(selected_lst[i])
    return selected_index


def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def find_index(target_lsa,selected_lst,max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa==target_lsa[i] and i not in selected_lst:
            return i
    return 0


def select_from_index(select_amount, indexlst):
    selected_index = []
    #print(indexlst)
    for i in range(select_amount):
        selected_index.append(indexlst[i])
    return selected_index


def build_neuron_tables(model, x_test, divide, output):
    total_num = x_test.shape[0]
    # init dict and its input
    neuron_interval = defaultdict(np.array)
    neuron_proba = defaultdict(np.array)
    layer = model.layers[-1]
    #test_output = build_testoutput(model, x_test)
    #output = test_output
    lower_bound = np.min(output, axis=0)
    upper_bound = np.max(output, axis=0)

    for index in range(output.shape[-1]):
        # compute interval
        # temp = (upper_bound[index] - lower_bound[index]) * .25
        # let interval = 30
        interval = np.linspace(
            lower_bound[index], upper_bound[index], divide)
        neuron_interval[(layer.name, index)] = interval
        neuron_proba[(layer.name, index)] = output_to_interval(
            output[:, index], interval) / total_num

    # save
    # np.save('neuron_interval.npy', neuron_interval)
    # np.save('neuron_proba.npy', neuron_proba)
    return neuron_interval, neuron_proba


def build_testoutput(model, x_test):
    input_tensor = model.input
    layer = model.layers[-1]
    # get this layer's output
    output = layer.output
    output_fun = K.function([input_tensor], [output])
    #print(output_fun)
    #output = output_fun([x_test])[0]

    N = 1000
    output = output_fun([x_test[0:N]])[0]
    #input_shape= x_test.shape[0]
    inputshape_N=int(x_test.shape[0]/N)
    for i in range(inputshape_N-1):
        tmpoutput = output_fun([x_test[N+i*N:2*N+i*N]])[0]
        #print(len(output))
        output = np.append(output, tmpoutput,axis=0)

    if inputshape_N*N!=x_test.shape[0]:
        tmpoutput = output_fun([x_test[inputshape_N*N:x_test.shape[0]]])[0]
        output = np.append(output, tmpoutput, axis=0)
    #print(len(output[0]))
    #output=output[0]

    output = output.reshape(output.shape[0], -1)
    #print(output[0])
    test_output = output
    return test_output

def neuron_entropy(model, neuron_interval, neuron_proba, sample_index,test_output):
    total_num = sample_index.shape[0]
    if(total_num == 0):
        return -1e3
    neuron_entropy = []
    layer = model.layers[-1]
    output = test_output
    output = output[sample_index, :]
    # get lower and upper bound of neuron output
    # lower_bound = np.min(output, axis=0)
    # upper_bound = np.max(output, axis=0)
    for index in range(output.shape[-1]):
        # compute interval
        #print('index:%d' % index)
        interval = neuron_interval[(layer.name, index)]
        bench_proba = neuron_proba[(layer.name, index)]
        test_proba = output_to_interval(
            output[:, index], interval) / total_num
        test_proba = np.clip(test_proba, 1e-10, 1 - 1e-10)
        log_proba = np.log(test_proba)
        temp_proba = bench_proba.copy()
        temp_proba[temp_proba < (.5 / total_num)] = 0
        entropy = np.sum(log_proba * temp_proba)
        neuron_entropy.append(entropy)
    return np.array(neuron_entropy)

def coverage(entropy):
    return np.mean(entropy)

def output_to_interval(output, interval):
    num = []
    for i in range(interval.shape[0] - 1):
        num.append(np.sum(np.logical_and(
            output > interval[i], output < interval[i + 1])))
    return np.array(num)


def selectsample(model, x_test, delta, iterate,neuron_interval,neuron_proba,test_output,attack=0):
    test = x_test
    #print(test)
    batch = delta

    max_index0 = np.random.choice(range(test.shape[0]), replace=False, size=30)
    for i in range(iterate):
        print('i:%d' % i)
        arr = np.random.permutation(test.shape[0])
        max_iter = 30
        e = neuron_entropy(model,neuron_interval,
                           neuron_proba, max_index0,test_output)
        cov = coverage(e)
        max_coverage = cov

        temp_cov = []
        index_list = []
        # select
        for j in range(max_iter):
            #print('j:%d' % j)
            #arr = np.random.permutation(test.shape[0])
            start = int(np.random.uniform(0, test.shape[0] - batch))
            #print(start)
            temp_index = np.append(max_index0, arr[start:start + batch])
            index_list.append(arr[start:start + batch])
            e = neuron_entropy(model, neuron_interval,
                               neuron_proba, temp_index,test_output)
            new_coverage = coverage(e)
            temp_cov.append(new_coverage)

        max_coverage = np.max(temp_cov)
        cov_index = np.argmax(temp_cov)
        max_index = index_list[cov_index]
        if(max_coverage <= cov):
            max_index = np.random.choice(range(test.shape[0]),replace=False,size=delta)
        max_index0 = np.append(max_index0, max_index)
        #if len(max_index0) in [100,300,500,1000]:
         #   tmpfile="./conditional/"+attack+"_svhn_"+str(len(max_index0))+".npy"
          #  np.save(tmpfile,max_index0)
           # print("saved!%s" %tmpfile)
    return max_index0


def conditional_sample(model, x_test, sample_size, attack=0):
    delta = 5
    iterate = int((sample_size - 30)/delta)
    print("build testoutput")
    test_output = build_testoutput(model, x_test)
    print("build neuron tables")
    neuron_interval, neuron_proba = build_neuron_tables(model, x_test, delta,test_output)
    print("selection")
    index_list = selectsample(model, x_test, delta, iterate, neuron_interval,neuron_proba,test_output,attack)
    #print(index_list)
    return list(index_list)


def CES_selection(model, candidate_data, select_size):
    CES_index = conditional_sample(model, candidate_data, select_size)
    select_index = select_from_index(select_size, CES_index)
    return select_index
