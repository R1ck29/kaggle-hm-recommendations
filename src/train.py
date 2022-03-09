import os; os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from tqdm import tqdm
import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k

# # Validation
def to_user_item_coo(df, ALL_USERS, ALL_ITEMS):
    """ Turn a dataframe with transactions into a COO sparse items x users matrix"""
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    return coo


def split_data(df, validation_days=7):
    """ Split a pandas dataframe into training and validation data, using <<validation_days>>
    """
    validation_cut = df['t_dat'].max() - pd.Timedelta(validation_days)

    df_train = df[df['t_dat'] < validation_cut]
    df_val = df[df['t_dat'] >= validation_cut]
    return df_train, df_val


def get_val_matrices(df, ALL_USERS, ALL_ITEMS, validation_days=7):
    """ Split into training and validation and create various matrices
        
        Returns a dictionary with the following keys:
            coo_train: training data in COO sparse format and as (users x items)
            csr_train: training data in CSR sparse format and as (users x items)
            csr_val:  validation data in CSR sparse format and as (users x items)
    
    """
    df_train, df_val = split_data(df, validation_days=validation_days)
    coo_train = to_user_item_coo(df_train, ALL_USERS, ALL_ITEMS)
    coo_val = to_user_item_coo(df_val, ALL_USERS, ALL_ITEMS)

    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()
    
    return {'coo_train': coo_train,
            'csr_train': csr_train,
            'csr_val': csr_val
          }


def validate(matrices, factors=200, iterations=20, regularization=0.01, show_progress=True):
    """ Train an ALS model with <<factors>> (embeddings dimension) 
    for <<iterations>> over matrices and validate with MAP@12
    """
    coo_train, csr_train, csr_val = matrices['coo_train'], matrices['csr_train'], matrices['csr_val']
    
    model = implicit.als.AlternatingLeastSquares(factors=factors, 
                                                 iterations=iterations, 
                                                 regularization=regularization, 
                                                 random_state=42)
    model.fit(coo_train, show_progress=show_progress)
    
    # The MAPK by implicit doesn't allow to calculate allowing repeated items, which is the case.
    # TODO: change MAP@12 to a library that allows repeated items in prediction
    map12 = mean_average_precision_at_k(model, csr_train, csr_val, K=12, show_progress=show_progress, num_threads=4)
    print(f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@12: {map12:6.5f}")
    return map12


def train(coo_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors, 
                                                 iterations=iterations, 
                                                 regularization=regularization, 
                                                 random_state=42)
    model.fit(coo_train, show_progress=show_progress)
    return model


# # Submission
# ## Submission function
def submit(model, csr_train, ALL_USERS, user_ids, item_ids, submission_name="submissions.csv"):
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USERS))
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        ids, scores = model.recommend(batch, csr_train[batch], N=12, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = user_ids[userid]
            user_items = ids[i]
            article_ids = [item_ids[item_id] for item_id in user_items]
            preds.append((customer_id, ' '.join(article_ids)))

    df_preds = pd.DataFrame(preds, columns=['customer_id', 'prediction'])
    df_preds.to_csv(submission_name, index=False)
    
    display(df_preds.head())
    print(df_preds.shape)
    
    return df_preds


def target_data(df, start_date, end_date):
    # Trying with less data:
    # https://www.kaggle.com/tomooinubushi/folk-of-time-is-our-best-friend/notebook
    df = df[(df['t_dat'] >= start_date) & (df['t_dat'] <= end_date)]
    s_date = str(df["t_dat"].min()).split(' ')[0]
    e_date = str(df['t_dat'].max()).split(' ')[0]
    print(f'Using date form {s_date} to {e_date}')
    return df


def train_als(df, dfu, dfi):
    # For validation this means 3 weeks of training and 1 week for validation
    # For submission, it means 4 weeks of training
    # ## Assign autoincrementing ids starting from 0 to both users and items
    ALL_USERS = dfu['customer_id'].unique().tolist()
    ALL_ITEMS = dfi['article_id'].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df['user_id'] = df['customer_id'].map(user_map)
    df['item_id'] = df['article_id'].map(item_map)

    del dfu, dfi

    # ## About CSR matrices
    # * https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    row = df['user_id'].values
    col = df['item_id'].values
    data = np.ones(df.shape[0])
    coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))

    # # Check that model works ok with data
    model = implicit.als.AlternatingLeastSquares(factors=10, iterations=2)
    model.fit(coo_train)

    matrices = get_val_matrices(df, ALL_USERS, ALL_ITEMS)

    best_map12 = 0
    print('Fiding best parameters...')
    for factors in tqdm([40, 50, 60, 100, 200, 500, 1000]):
        for iterations in [3, 12, 14, 15, 20]:
            for regularization in [0.01]:
                map12 = validate(matrices, factors, iterations, regularization, show_progress=False)
                if map12 > best_map12:
                    best_map12 = map12
                    best_params = {'factors': factors, 'iterations': iterations, 'regularization': regularization}
                    print(f"Best MAP@12 found. {best_map12:6.5f} Updating: {best_params}")

    # Factors: 100 - Iterations: 12 - Regularization: 0.010 ==> MAP@12: 0.00638
    # Best MAP@12 found. Updating: {'factors': 100, 'iterations': 12, 'regularization': 0.01}
    del matrices

    # # Training over the full dataset
    coo_train = to_user_item_coo(df)
    csr_train = coo_train.tocsr()

    print(f'best_params: {best_params}')
    model = train(coo_train, **best_params)

    submission_name = model_dir + "/submission_" + str(best_map12) + ".csv"
    df_preds = submit(model, csr_train, ALL_USERS, user_ids, item_ids, submission_name)


if __name__ == '__main__':
    # # Load dataframes
    base_path = '../input/h-and-m-personalized-fashion-recommendations/'
    csv_train = f'{base_path}transactions_train.csv'
    csv_sub = f'{base_path}sample_submission.csv'
    csv_users = f'{base_path}customers.csv'
    csv_items = f'{base_path}articles.csv'

    df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
    df_sub = pd.read_csv(csv_sub)
    dfu = pd.read_csv(csv_users)
    dfi = pd.read_csv(csv_items, dtype={'article_id': str})

    exp_name = 'exp1'
    model_name = 'ALS'
    start_date = '2020-01-21' #'2020-08-21'
    end_date =  str(df['t_dat'].max()).split(' ')[0]

    model_dir = f'../models/{exp_name}'
    os.makedirs(model_dir, exist_ok=True)

    df = target_data(df, start_date, end_date)

    if model_name == 'ALS':
        train_als(df, dfu, dfi)