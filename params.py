EMB_DIM = 32
BATCH_SIZE_1 = 128
BATCH_SIZE_2 = 1024
BATCH_SIZE = 1024
DECAY = 0.001
LAMDA = 1
K = 2
N_EPOCH = 200
DROPOUT = 0.1
LR = 0.001
# Determine how many epochs to train with a test
Test_Every = 1


# Dataset: 'Amazon'/'Douban'
DATASET = 'Amazon'
# choose two related domains in the dataset you choose
# metaList: Amazon-Movie-Music/Amazon-Cell-Elec/Douban-Movie-Book/Douban-Music-Book
metaName_1 = 'Cell'
metaName_2 = 'Elec'

DIR = '/data/' + DATASET + '/' + metaName_1 + '_' + metaName_2 + '/'
MODEL_DIR = '/trained_model/' + DATASET + '_' + metaName_1 + '_' + metaName_2 + '/'

eigen_pickle_file_1 = DIR + metaName_1 + '_eigenPickle.pickle'
eigen_pickle_file_2 = DIR + metaName_2 + '_eigenPickle.pickle'

adjacent_matrix_1 = DIR + metaName_1 + '_adjacent_matrix.pickle'
adjacent_matrix_2 = DIR + metaName_2 + '_adjacent_matrix.pickle'
domain_adjacent_matrix = DIR + 'domain_matrix.pickle'

item_adjacent_path = DIR + 'item_adjacent_matrix.pickle'
cross_domain_item_adjacent_path = DIR + metaName_1 + '_' + metaName_2 + '_item_adj.pickle'

cross_adjacent_matrix = DIR + metaName_1 + '_' + metaName_2 + '_cross_adjacent_matrix.pickle'

if DATASET == 'Amazon':
    filepath_1 = DIR + metaName_1 + '_all_item_list.dat'
    filepath_2 = DIR + metaName_2 + '_all_item_list.dat'
    
    adjacent_matrix_1 = DIR + metaName_1 + '_adjacent_matrix.pickle'
    adjacent_matrix_2 = DIR + metaName_2 + '_adjacent_matrix.pickle'
    domain_adjacent_matrix = DIR + 'domain_matrix.pickle'

    item_adjacent_path = DIR + 'item_adjacent_matrix.pickle'
    cross_domain_item_adjacent_path = DIR + metaName_1 + '_' + metaName_2 + '_item_adj.pickle'
    cross_adjacent_matrix = DIR + metaName_1 + '_' + metaName_2 + '_cross_adjacent_matrix.pickle'

if DATASET == 'Douban':
    filepath_1 = DIR + metaName_1 + '_all_item_list.dat'
    filepath_2 = DIR + metaName_2 + '_all_item_list.dat'

    adjacent_matrix_1 = DIR + metaName_1 + '_adjacent_matrix.pickle'
    adjacent_matrix_2 = DIR + metaName_2 + '_adjacent_matrix.pickle'
    domain_adjacent_matrix = DIR + 'domain_matrix.pickle'

    item_adjacent_path = DIR + 'item_adjacent_matrix.pickle'
    cross_domain_item_adjacent_path = DIR + metaName_1 + '_' + metaName_2 + '_item_adj.pickle'

    cross_adjacent_matrix = DIR + metaName_1 + '_' + metaName_2 + '_cross_adjacent_matrix.pickle'