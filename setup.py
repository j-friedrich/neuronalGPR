import io
import os
import requests
import zipfile

# download the snelson dataset
if not os.path.isdir('snelson_data/'):
    r = requests.get('http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip', stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extract('SPGP_dist/train_inputs')
    z.extract('SPGP_dist/train_outputs')
    os.system('mv SPGP_dist snelson_data')


# download datasets from Yarin's repo
_UCI_DIRECTORY_PATH = "DropoutUncertaintyExps-master/UCI_Datasets/"
if not os.path.isdir(_UCI_DIRECTORY_PATH):
    r = requests.get(
        'https://github.com/yaringal/DropoutUncertaintyExps/archive/master.zip', stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


if not os.path.isdir(_UCI_DIRECTORY_PATH + 'year-prediction-MSD/data/'):
    # download MSD dataset that's missing in Yarin's repo
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/' +
                     '00203/YearPredictionMSD.txt.zip', stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

    # create files in Yarin's format
    os.makedirs(_UCI_DIRECTORY_PATH + 'year-prediction-MSD/data/')

    for (fname, low, up) in (('index_train_0.txt', 0, 463715),
                             ('index_test_0.txt', 463715, 515345),
                             ('index_features.txt', 1, 91)):
        with open(_UCI_DIRECTORY_PATH + 'year-prediction-MSD/data/' + fname, 'w') as f:
            for n in range(low, up):
                f.write(str(n) + '\n')

    for (fname, n) in (('n_epochs.txt', 40), ('n_splits.txt', 1),
                       ('n_hidden.txt', 100), ('index_target.txt', 0)):
        with open(_UCI_DIRECTORY_PATH + 'year-prediction-MSD/data/' + fname, 'w') as f:
            f.write(str(n) + '\n')

    with open('YearPredictionMSD.txt', 'r') as fin, open(
            _UCI_DIRECTORY_PATH + 'year-prediction-MSD/data/data.txt', 'w') as fout:
        for line in fin:
            fout.write(line.replace(',', ' '))


# download PBP from Miguel's repo
if not os.path.isdir('Probabilistic-Backpropagation-master/'):
    r = requests.get(
        'https://github.com/HIPS/Probabilistic-Backpropagation/archive/master.zip', stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    os.system('cd Probabilistic-Backpropagation-master/c/PBP_net ; chmod +x compile.sh ; ./compile.sh')


# create results directories if not existent yet
for method in ('GP', 'VFE', 'FITC', 'BioNN', 'ANN', 'Dropout', 'PBP'):
    os.makedirs('results/' + method, exist_ok=True)
os.makedirs('fig', exist_ok=True)
