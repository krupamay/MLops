from utils import get_all_hyper_params


def test_hparam_cominations ():
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params={}
    h_params ['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_all_hyper_params(gamma_list, C_list)
    print(h_params_combinations)
    assert len(h_params_combinations) == 16
    assert len (h_params_combinations) == len(gamma_list) * len(C_list)

test_hparam_cominations()
def test_hparam_combination_exists():
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    h_params={}
    h_params ['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_all_hyper_params(gamma_list, C_list)


    assert {'gamma': 0.001, 'C': 10} in h_params_combinations