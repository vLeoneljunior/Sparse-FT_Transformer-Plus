"Code from https://github.com/google-research/google-research/blob/master/tabnet/experiment_covertype.py"
# %%
from pathlib import Path

import numpy as np
import tensorflow as tf
import zero

from rtdl_revisiting_models import lib


# %%
def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class TabNet:
    """TabNet model class."""

    def __init__(
        self,
        num_features,
        columns,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        batch_momentum,
        virtual_batch_size,
        num_classes,
        epsilon=0.00001,
        **kwargs,
    ):
        """Initializes a TabNet instance.
        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.
        Returns:
          A TabNet instance.
        """

        self.columns = columns
        self.num_features = num_features
        self.num_classes = num_classes

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

    def encoder(self, data, reuse, is_training):
        """TabNet encoder model."""

        with tf.name_scope("Encoder"):

            # Reads and normalizes input features.
            # NOTE we do data normalization at a dataset level
            if self.columns:
                # TensorFlow 2.x replacement for tf.feature_column.input_layer
                if not hasattr(self, '_feature_layer'):
                    self._feature_layer = tf.keras.utils.DenseFeatures(self.columns)
                features = self._feature_layer(data)
            else:
                features = data

            # features = tf.layers.batch_normalization(
            #     features, training=is_training, momentum=self.batch_momentum
            # )
            batch_size = tf.shape(features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complemantary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features]
            )
            total_entropy = 0

            if is_training:
                v_b = self.virtual_batch_size
            else:
                v_b = 1

            for ni in range(self.num_decision_steps):

                # Feature transformer with two shared and two decision step dependent
                # blocks is used below.

                reuse_flag = ni > 0

                with tf.name_scope(f"transform_f1"):
                    if reuse_flag and hasattr(self, '_transform_f1_layer'):
                        transform_f1 = self._transform_f1_layer(masked_features)
                    else:
                        self._transform_f1_layer = tf.keras.layers.Dense(
                            self.feature_dim * 2,
                            use_bias=False,
                            name="Transform_f1"
                        )
                        transform_f1 = self._transform_f1_layer(masked_features)
                    
                    transform_f1 = self._ghost_batch_norm(
                        transform_f1, v_b, is_training, 
                        name=f"transform_f1_bn_{ni}", reuse=reuse_flag
                    )
                    transform_f1 = glu(transform_f1, self.feature_dim)

                with tf.name_scope(f"transform_f2"):
                    if reuse_flag and hasattr(self, '_transform_f2_layer'):
                        transform_f2 = self._transform_f2_layer(transform_f1)
                    else:
                        self._transform_f2_layer = tf.keras.layers.Dense(
                            self.feature_dim * 2,
                            use_bias=False,
                            name="Transform_f2"
                        )
                        transform_f2 = self._transform_f2_layer(transform_f1)
                    
                    transform_f2 = self._ghost_batch_norm(
                        transform_f2, v_b, is_training,
                        name=f"transform_f2_bn_{ni}", reuse=reuse_flag
                    )
                    transform_f2 = (
                        glu(transform_f2, self.feature_dim) + transform_f1
                    ) * np.sqrt(0.5)

                with tf.name_scope(f"transform_f3_{ni}"):
                    if not hasattr(self, f'_transform_f3_layer_{ni}'):
                        setattr(self, f'_transform_f3_layer_{ni}', 
                               tf.keras.layers.Dense(
                                   self.feature_dim * 2,
                                   use_bias=False,
                                   name=f"Transform_f3_{ni}"
                               ))
                    transform_f3_layer = getattr(self, f'_transform_f3_layer_{ni}')
                    transform_f3 = transform_f3_layer(transform_f2)
                    
                    transform_f3 = self._ghost_batch_norm(
                        transform_f3, v_b, is_training,
                        name=f"transform_f3_bn_{ni}", reuse=False
                    )
                    transform_f3 = (
                        glu(transform_f3, self.feature_dim) + transform_f2
                    ) * np.sqrt(0.5)

                with tf.name_scope(f"transform_f4_{ni}"):
                    if not hasattr(self, f'_transform_f4_layer_{ni}'):
                        setattr(self, f'_transform_f4_layer_{ni}',
                               tf.keras.layers.Dense(
                                   self.feature_dim * 2,
                                   use_bias=False,
                                   name=f"Transform_f4_{ni}"
                               ))
                    transform_f4_layer = getattr(self, f'_transform_f4_layer_{ni}')
                    transform_f4 = transform_f4_layer(transform_f3)
                    
                    transform_f4 = self._ghost_batch_norm(
                        transform_f4, v_b, is_training,
                        name=f"transform_f4_bn_{ni}", reuse=False
                    )
                    transform_f4 = (
                        glu(transform_f4, self.feature_dim) + transform_f3
                    ) * np.sqrt(0.5)

                if ni > 0:

                    decision_out = tf.nn.relu(transform_f4[:, : self.output_dim])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True) / (
                        self.num_decision_steps - 1
                    )
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = transform_f4[:, self.output_dim :]

                if ni < self.num_decision_steps - 1:

                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    with tf.name_scope(f"transform_coef_{ni}"):
                        if not hasattr(self, f'_transform_coef_layer_{ni}'):
                            setattr(self, f'_transform_coef_layer_{ni}',
                                   tf.keras.layers.Dense(
                                       self.num_features,
                                       use_bias=False,
                                       name=f"Transform_coef_{ni}"
                                   ))
                        coef_layer = getattr(self, f'_transform_coef_layer_{ni}')
                        mask_values = coef_layer(features_for_coef)
                        
                        mask_values = self._ghost_batch_norm(
                            mask_values, v_b, is_training,
                            name=f"transform_coef_bn_{ni}", reuse=False
                        )
                    
                    mask_values *= complemantary_aggregated_mask_values
                    mask_values = self._sparsemax(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complemantary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values
                    )

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(
                            -mask_values * tf.math.log(mask_values + self.epsilon), axis=1
                        )
                    ) / (self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, features)

                    # Visualization of the feature selection mask at decision step ni
                    tf.summary.image(
                        f"Mask_for_step_{ni}",
                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                        max_outputs=1,
                    )

            # Visualization of the aggregated feature importances
            tf.summary.image(
                "Aggregated_mask",
                tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
                max_outputs=1,
            )

            return output_aggregated, total_entropy

    def classify(self, activations, reuse):
        """TabNet classify block."""

        with tf.name_scope("Classify"):
            if not hasattr(self, '_classify_layer'):
                self._classify_layer = tf.keras.layers.Dense(
                    self.num_classes, 
                    use_bias=False,
                    name="classify_dense"
                )
            logits = self._classify_layer(activations)
            predictions = tf.nn.softmax(logits)
            return logits, predictions

    def regress(self, activations, reuse):
        """TabNet regress block."""

        with tf.name_scope("Regress"):
            if not hasattr(self, '_regress_layer'):
                self._regress_layer = tf.keras.layers.Dense(
                    1,
                    name="regress_dense"
                )
            predictions = self._regress_layer(activations)
            return predictions

    def _ghost_batch_norm(self, inputs, virtual_batch_size, training, name, reuse=False):
        """Ghost batch normalization implementation."""
        if not hasattr(self, f'_bn_layers'):
            self._bn_layers = {}
        
        if name not in self._bn_layers:
            self._bn_layers[name] = tf.keras.layers.BatchNormalization(
                momentum=self.batch_momentum,
                name=name
            )
        
        bn_layer = self._bn_layers[name]
        
        if training and virtual_batch_size > 1:
            # Apply ghost batch normalization
            batch_size = tf.shape(inputs)[0]
            # Ensure virtual batch size doesn't exceed actual batch size
            v_b = tf.minimum(virtual_batch_size, batch_size)
            
            if batch_size >= v_b:
                # Split into chunks for ghost batch norm
                chunks = tf.split(inputs, batch_size // v_b, axis=0)
                normalized_chunks = []
                for chunk in chunks:
                    normalized_chunk = bn_layer(chunk, training=training)
                    normalized_chunks.append(normalized_chunk)
                return tf.concat(normalized_chunks, axis=0)
            else:
                return bn_layer(inputs, training=training)
        else:
            return bn_layer(inputs, training=training)

    def _sparsemax(self, logits):
        """Sparsemax activation function using TensorFlow Addons."""
        import tensorflow_addons as tfa
        return tfa.activations.sparsemax(logits)


def make_tf_loaders(args, X, Y):
    """Create TensorFlow data loaders from numpy arrays."""
    datasets = {k: tf.data.Dataset.from_tensor_slices((X[k], Y[k])) for k in X.keys()}
    X_loader = {}
    Y_loader = {}

    for k in datasets.keys():
        if k == lib.TRAIN:
            datasets[k] = datasets[k].shuffle(
                buffer_size=50, reshuffle_each_iteration=True
            )
            datasets[k] = datasets[k].batch(
                args["training"]["batch_size"], drop_remainder=True
            )
        else:
            datasets[k] = datasets[k].batch(args["training"]["batch_size"])

        # NOTE +1 for the final validation step for the best model at the end
        datasets[k] = datasets[k].repeat(args["training"]["epochs"] + 1)
        datasets[k] = datasets[k].as_numpy_iterator()

    # Add train with no shuffle dataset for final eval
    ds = tf.data.Dataset.from_tensor_slices((X[lib.TRAIN], Y[lib.TRAIN]))
    ds = ds.batch(args["training"]["batch_size"])
    ds = ds.as_numpy_iterator()
    k = "train_noshuffle"
    datasets[k] = ds

    return datasets, X_loader, Y_loader


# %%
def get_train_eval_ops(args, data: lib.Dataset, model):
    """Create train step function and prediction functions"""
    
    # Setup optimizer with learning rate schedule
    initial_lr = args["training"]["schedule"]["learning_rate"]
    decay_steps = args["training"]["schedule"]["decay_steps"] 
    decay_rate = args["training"]["schedule"]["decay_rate"]
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    @tf.function
    def train_step(x_batch, y_batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            encoder_out, total_entropy = model.encoder(x_batch, reuse=False, is_training=True)
            
            # Compute predictions and loss based on task type
            if data.is_multiclass:
                y_pred, _ = model.classify(encoder_out, reuse=False)
                cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
                    y_batch, y_pred, from_logits=True
                )
                loss = tf.reduce_mean(cross_entropy) + args["training"]["sparsity_loss_weight"] * total_entropy
            elif data.is_regression:
                y_pred = model.regress(encoder_out, reuse=False)
                mse = tf.keras.losses.mean_squared_error(
                    tf.expand_dims(y_batch, axis=1), y_pred
                )
                loss = tf.reduce_mean(mse) + args["training"]["sparsity_loss_weight"] * total_entropy
            else:  # binary classification
                y_pred = model.regress(encoder_out, reuse=False)
                log_loss = tf.keras.losses.binary_crossentropy(
                    tf.expand_dims(y_batch, axis=1),
                    tf.nn.sigmoid(y_pred),
                    from_logits=False
                )
                loss = tf.reduce_mean(log_loss) + args["training"]["sparsity_loss_weight"] * total_entropy
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Gradient clipping
        grad_thresh = args["training"]["grad_thresh"]
        gradients = [tf.clip_by_value(grad, -grad_thresh, grad_thresh) for grad in gradients]
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss

    @tf.function
    def predict_step(x_batch, split_name):
        """Prediction step for evaluation"""
        encoder_out, _ = model.encoder(x_batch, reuse=True, is_training=False)
        
        if data.is_multiclass:
            y_pred, _ = model.classify(encoder_out, reuse=True)
            return y_pred
        else:
            y_pred = model.regress(encoder_out, reuse=True)
            return y_pred

    @property
    def trainable_variables(self):
        """Get all trainable variables from the model"""
        variables = []
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'trainable_variables'):
                variables.extend(attr.trainable_variables)
            elif isinstance(attr, dict):
                for layer in attr.values():
                    if hasattr(layer, 'trainable_variables'):
                        variables.extend(layer.trainable_variables)
        return variables
    
    # Add trainable_variables property to model
    model.trainable_variables = trainable_variables
    
    return train_step, predict_step


def evaluate(args, y, datasets, predict_step, parts, task_type, y_info):
    """Evaluate model performance on given dataset parts"""
    metrics = {}
    predictions = {}

    for part in parts:
        y_pred_list = []
        dataset_iter = iter(datasets[part])
        
        # Calculate number of batches
        batch_size = args["training"]["batch_size"]
        num_samples = y[part].shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            try:
                x_batch, _ = next(dataset_iter)
                y_pred_batch = predict_step(x_batch, part)
                y_pred_list.append(y_pred_batch.numpy())
            except StopIteration:
                break
        
        if y_pred_list:
            y_pred = np.concatenate(y_pred_list)
            # Trim predictions to match target size
            y_pred = y_pred[:num_samples]
            
            metrics[part] = lib.calculate_metrics(
                task_type, y[part], y_pred, 'logits', y_info
            )
            predictions[part] = y_pred
        else:
            print(f"Warning: No predictions generated for {part}")
            metrics[part] = {'score': 0.0}
            predictions[part] = np.array([])

    for part, part_metrics in metrics.items():
        print(f'[{part:<5}]', lib.make_summary(part_metrics))

    return metrics, predictions


# %%
args, output = lib.load_config()
try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass
dataset_dir = lib.get_path(args['data']['path'])
stats = {
    "dataset": args["data"]["path"],
    "algorithm": Path(__file__).stem,
    **lib.load_json(output / "stats.json"),
}

# Clear any existing default graph and set random seeds
tf.keras.backend.clear_session()
tf.random.set_seed(args["seed"])

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'indices'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
if not isinstance(X, tuple):
    X = (X, None)

try:
    zero.random.seed(args['seed'])
except AttributeError:
    try:
        zero.set_randomness(args['seed'])
    except AttributeError:
        pass
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')
X_num, X_cat = X

use_placeholders = D.info["name"] in ["epsilon", "yahoo"]
columns = None

if use_placeholders:
    X = X_num
    # For epsilon dataset, use direct numpy arrays
    pass
else:
    if X_cat is not None:
        X = {}
        for part in lib.PARTS:
            X[part] = {}
            for i in range(X_num[part].shape[1]):
                X[part][str(i)] = X_num[part][:, i]
            for i in range(
                X_num[part].shape[1], X_num[part].shape[1] + X_cat[part].shape[1]
            ):
                X[part][str(i)] = X_cat[part][:, i - X_num[part].shape[1]]
    else:
        X = X_num

    datasets_tf, x_loader, y_loader = make_tf_loaders(args, X, Y)

    if X_cat is not None:
        num_columns = [
            tf.feature_column.numeric_column(str(i))
            for i in range(X_num['train'].shape[1])
        ]
        cat_columns = [
            tf.feature_column.categorical_column_with_identity(
                str(i), max(X_cat['train'][:, i - X_num['train'].shape[1]]) + 1
            )
            for i in range(
                X_num['train'].shape[1],
                X_num['train'].shape[1] + X_cat['train'].shape[1],
            )
        ]
        emb_columns = [
            tf.feature_column.embedding_column(c, args["model"]["d_embedding"])
            for c in cat_columns
        ]
        columns = num_columns + emb_columns

# Restricting hyperparameter search space from original paper
# 1. N_a = N_b

args["model"]["output_dim"] = args["model"]["feature_dim"]
print(columns)

model = TabNet(
    num_classes=D.info['n_classes'] if D.is_multiclass else 1,
    columns=columns,
    num_features=X_num['train'].shape[1]
    + (0 if X_cat is None else args["model"]["d_embedding"] * X_cat['train'].shape[1]),
    **args["model"],
)

train_step, predict_step = get_train_eval_ops(args, D, model)

batch_size = stats['batch_size'] = args["training"]["batch_size"]
epoch_size = stats['epoch_size'] = (
    Y[lib.TRAIN].shape[0] // batch_size
)  # drop_last=True in tf
progress = zero.ProgressTracker(args["training"]["patience"])

timer = zero.Timer()
timer.run()

# Training loop
for e in range(args["training"]["epochs"]):
    epoch_timer = zero.Timer()
    epoch_timer.run()

    if use_placeholders:
        loader = lib.IndexLoader(Y[lib.TRAIN].shape[0], batch_size, True, "cpu")
        loader = iter(loader)

        for step in range(epoch_size):
            idx = next(loader)
            x_batch = X[lib.TRAIN][idx]
            y_batch = Y[lib.TRAIN][idx]

            if step % args["training"]["display_steps"] == 0:
                train_loss = train_step(x_batch, y_batch)
                print(f"Step {step}, Train Loss {train_loss:.4f}")
            else:
                _ = train_step(x_batch, y_batch)
    else:
        # Use tf.data datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X[lib.TRAIN], Y[lib.TRAIN]))
        train_dataset = train_dataset.shuffle(buffer_size=50, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        
        for step, (x_batch, y_batch) in enumerate(train_dataset.take(epoch_size)):
            if step % args["training"]["display_steps"] == 0:
                train_loss = train_step(x_batch, y_batch)
                print(f"Step {step}, Train Loss {train_loss:.4f}")
            else:
                _ = train_step(x_batch, y_batch)

    print(f"Epoch {e} done; time {zero.format_seconds(epoch_timer())}")
    
    # Evaluation
    if use_placeholders:
        # Create temporary datasets for evaluation
        eval_datasets = {}
        for part in [lib.VAL, lib.TEST]:
            dataset = tf.data.Dataset.from_tensor_slices((X[part], Y[part]))
            dataset = dataset.batch(batch_size)
            eval_datasets[part] = dataset
        
        metrics, predictions = evaluate(
            args, Y, eval_datasets, predict_step, [lib.VAL, lib.TEST], D.info['task_type'], y_info
        )
    else:
        # Create evaluation datasets
        eval_datasets = {}
        for part in [lib.VAL, lib.TEST]:
            dataset = tf.data.Dataset.from_tensor_slices((X[part], Y[part]))
            dataset = dataset.batch(batch_size)
            eval_datasets[part] = dataset
        
        metrics, predictions = evaluate(
            args, Y, eval_datasets, predict_step, [lib.VAL, lib.TEST], D.info['task_type'], y_info
        )
    
    progress.update(metrics[lib.VAL]["score"])

    if progress.success:
        print("New best epoch")
        stats["best_epoch"] = e
        # Save model weights
        model_weights = {}
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'get_weights'):
                model_weights[attr_name] = attr.get_weights()
            elif isinstance(attr, dict):
                for layer_name, layer in attr.items():
                    if hasattr(layer, 'get_weights'):
                        model_weights[f"{attr_name}_{layer_name}"] = layer.get_weights()
        
        # Save using numpy
        np.savez(str(output / "best_weights.npz"), **{k: np.array(v) for k, v in model_weights.items() if v})
        lib.dump_stats(stats, output, final=False)
    elif progress.fail:
        print("Early stopping")
        break

# Final evaluation with all parts
if use_placeholders:
    eval_datasets = {}
    for part in lib.PARTS:
        dataset = tf.data.Dataset.from_tensor_slices((X[part], Y[part]))
        dataset = dataset.batch(batch_size)
        eval_datasets[part] = dataset
    
    stats['metrics'], predictions = evaluate(
        args, Y, eval_datasets, predict_step, lib.PARTS, D.info['task_type'], y_info
    )
else:
    eval_datasets = {}
    for part in lib.PARTS:
        dataset = tf.data.Dataset.from_tensor_slices((X[part], Y[part]))
        dataset = dataset.batch(batch_size)
        eval_datasets[part] = dataset
    
    stats['metrics'], predictions = evaluate(
        args, Y, eval_datasets, predict_step, lib.PARTS, D.info['task_type'], y_info
    )

for k, v in predictions.items():
    np.save(output / f'p_{k}.npy', v)

stats['time'] = zero.format_seconds(timer())
lib.dump_stats(stats, output, final=True)

print(f"Total time: {zero.format_seconds(timer())}")