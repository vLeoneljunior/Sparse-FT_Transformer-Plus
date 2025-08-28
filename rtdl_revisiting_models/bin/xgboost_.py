from copy import deepcopy
from pathlib import Path
import os
import warnings

import numpy as np
import zero
from xgboost import XGBClassifier, XGBRegressor

from rtdl_revisiting_models import lib

# Suppress warnings
warnings.filterwarnings("ignore", message=".*suggest_loguniform.*deprecated.*")
warnings.filterwarnings("ignore", message=".*suggest_uniform.*deprecated.*")
warnings.filterwarnings("ignore", message=".*sparse.*renamed.*sparse_output.*")
warnings.filterwarnings("ignore", message=".*eval_metric.*deprecated.*")
warnings.filterwarnings("ignore", message=".*early_stopping_rounds.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

args, output = lib.load_config()
args['model']['random_state'] = args['seed']

try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass

dataset_dir = lib.get_path(args['data']['path'])
stats = lib.load_json(output / 'stats.json')
stats.update({'dataset': dataset_dir.name, 'algorithm': Path(__file__).stem})

# Prepare data and model
D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
assert isinstance(X, dict)

try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass

Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

# Prepare model kwargs
fit_kwargs = deepcopy(args["fit"])
model_kwargs = deepcopy(args["model"])

# Extract early_stopping_rounds
early_stopping_rounds = fit_kwargs.pop('early_stopping_rounds', None)
eval_set = [(X[lib.VAL], Y[lib.VAL])]

# GPU configuration - SIMPLE APPROACH
# Let XGBoost handle device mismatch internally
if 'tree_method' in model_kwargs and 'gpu' in str(model_kwargs.get('tree_method')).lower():
    model_kwargs['tree_method'] = 'hist'
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        model_kwargs['device'] = 'cuda'
        print("XGBoost configured for GPU training (will handle data transfer internally)")
    else:
        model_kwargs['device'] = 'cpu'
        print("XGBoost configured for CPU training")

# Create model
if D.is_regression:
    model = XGBRegressor(
        **model_kwargs,
        early_stopping_rounds=early_stopping_rounds
    )
    predict = model.predict
    eval_metric = 'rmse'
else:
    if D.is_multiclass:
        eval_metric = 'merror'
        predict = lambda model, x: model.predict_proba(x)
    else:
        eval_metric = 'error'
        predict = lambda model, x: model.predict_proba(x)[:, 1]
    
    model = XGBClassifier(
        **model_kwargs,
        disable_default_eval_metric=True,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric=eval_metric
    )

# Fit model
timer = zero.Timer()
timer.run()
print("Training XGBoost...")
model.fit(X[lib.TRAIN], Y[lib.TRAIN], eval_set=eval_set, verbose=False)

# Save model and metrics
model.save_model(str(output / "model.json"))
np.save(output / "feature_importances.npy", model.feature_importances_)

stats['metrics'] = {}
for part in X:
    if D.is_regression:
        p = predict(X[part])
    else:
        p = predict(model, X[part])
    
    stats['metrics'][part] = lib.calculate_metrics(
        D.info['task_type'], Y[part], p, 'probs', y_info
    )
    np.save(output / f'p_{part}.npy', p)

stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)