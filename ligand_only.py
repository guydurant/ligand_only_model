import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
import os
from sklearn.ensemble import RandomForestRegressor
import pickle


def get_rdkit_features_names():
    with open('rdkit_feature_names.txt', 'r') as f:
        return f.read().splitlines()


def generate_rdkit_features(csv_file, data_dir):
    feature_names = get_rdkit_features_names()
    descriptors = {d[0]: d[1] for d in Descriptors.descList}
    print(len(feature_names), 'feature names')
    print(len(descriptors), 'descriptors')
    _, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    features = {}
    for i in tqdm(ligand_files):
        mol = Chem.MolFromMolFile(i)
        features[keys[ligand_files.index(i)]] = [descriptors[f](mol) for f in feature_names]
        # features[keys[ligand_files.index(i)].append(pks[ligand_files.index(i)])]
    columns = feature_names
    # print(features)
    return pd.DataFrame.from_dict(features, orient='index', columns=columns)


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [os.path.join(data_dir, protein_file) for protein_file in df['protein']]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df['ligand']]
    keys = df['key']
    pks = df['pk']
    return protein_files, ligand_files, keys, pks


def train_model(csv_file):
    _, _, _, pks = load_csv(csv_file, args.data_dir)
    features_df = pd.read_csv(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv', index_col=0)
    print(features_df.shape)
    model = RandomForestRegressor(n_estimators=500, max_features='sqrt', max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1)
    model.fit(features_df, pks)
    return model

def predict(model, csv_file, data_dir):
    _, _, keys, pks = load_csv(csv_file, data_dir)
    features_df = pd.read_csv(f'temp_features/{csv_file.split("/")[-1].split(".")[0]}_features.csv', index_col=0)
    predictions = model.predict(features_df)
    results_df = pd.DataFrame({'key': keys, 'pred': predictions, 'pk': pks})
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='train.csv')
    parser.add_argument('--val_csv_file', type=str, default='val.csv')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--val_data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    # data = pd.read_csv('temp_features/pdbbind_2020_general_crystal_all_features.csv')
    # print(data.shape)
    if args.train:
        if not os.path.exists(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'):
            df = generate_rdkit_features(args.csv_file, args.data_dir)
            df.to_csv(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv')
        model = train_model(args.csv_file)
        with open(f'temp_models/{args.model_name}.pkl', 'wb') as handle:
            pickle.dump(model, handle)

    elif args.predict:
        with open(f'temp_models/{args.model_name}.pkl', 'rb') as handle:
            model = pickle.load(handle)
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'):
            df = generate_rdkit_features(args.val_csv_file, args.val_data_dir)
            df.to_csv(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv')
        results_df = predict(model, args.val_csv_file, args.val_data_dir)
        results_df.to_csv(f'results/{args.model_name}_{args.val_csv_file.split("/")[-1]}', index=False)
    else:
        raise ValueError('Need to define mode, --train or --predict')

