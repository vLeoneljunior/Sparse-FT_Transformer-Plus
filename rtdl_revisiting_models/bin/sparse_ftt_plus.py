# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor

import lib
from sparse_ftt_plus import InterpretableFTTPlus


# %%
if __name__ == "__main__":
    args, output = lib.load_config()
    # defaults specific to sparse FTT+
    args['model'].setdefault('attention_dropout', 0.0)
    args['model'].setdefault('residual_dropout', 0.0)
    args['model'].setdefault('ffn_dropout', 0.0)
    args['model'].setdefault('attention_initialization', 'kaiming')
    args['model'].setdefault('attention_normalization', 'LayerNorm')
    args['model'].setdefault('ffn_activation', 'ReGLU')
    args['model'].setdefault('ffn_normalization', 'LayerNorm')
    args['model'].setdefault('prenormalization', True)
    args['model'].setdefault('head_activation', 'ReLU')
    args['model'].setdefault('head_normalization', 'LayerNorm')
    args['model'].setdefault('n_heads', 8)

    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

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
    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    del X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )

    # instantiate InterpretableFTTPlus
    n_num = 0 if X_num is None else X_num['train'].shape[1]
    d_out = D.info['n_classes'] if D.is_multiclass else 1

    model = InterpretableFTTPlus.make_baseline(
        n_num_features=n_num,
        d_token=args['model'].get('d_token'),
        n_blocks=args['model'].get('n_blocks'),
        n_heads=args['model'].get('n_heads'),
        attention_dropout=args['model'].get('attention_dropout'),
        ffn_d_hidden=args['model'].get('ffn_d_hidden'),
        ffn_dropout=args['model'].get('ffn_dropout'),
        residual_dropout=args['model'].get('residual_dropout'),
        d_out=d_out,
        attention_initialization=args['model'].get('attention_initialization'),
        attention_normalization=args['model'].get('attention_normalization'),
        ffn_activation=args['model'].get('ffn_activation'),
        ffn_normalization=args['model'].get('ffn_normalization'),
        prenormalization=args['model'].get('prenormalization'),
        head_activation=args['model'].get('head_activation'),
        head_normalization=args['model'].get('head_normalization'),
        num_tokenizer=None,
    ).to(device)
    
    # If categorical features are present, attach a categorical tokenizer so both
    # numerical and categorical features are handled.
    if X_cat is not None:
        categories = lib.get_categories(X_cat)
        if categories:
            from rtdl_lib.modules import CategoricalFeatureTokenizer
            # use the same d_token / initialization as the numerical tokenizer
            model.feature_tokenizer.cat_tokenizer = CategoricalFeatureTokenizer(
                categories,
                model.feature_tokenizer.d_token,
                True,
                model.feature_tokenizer.initialization,
            )
            model.feature_tokenizer.cat_tokenizer.to(device)
    
    if torch.cuda.device_count() > 1:
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = lib.get_n_parameters(model)

    # optimizer param groups from model if available
    if hasattr(model, 'optimization_param_groups'):
        param_groups = model.optimization_param_groups()
    else:
        def needs_wd(name):
            return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

        parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
        param_groups = [{'params': parameters_with_wd}, {'params': parameters_without_wd, 'weight_decay': 0.0}]

    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        param_groups,
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    def apply_model(part, idx):
        # Support both numerical and categorical inputs. If model was wrapped with
        # nn.DataParallel, access the underlying module.
        real_model = model.module if isinstance(model, nn.DataParallel) else model
        x_num_batch = None if X_num is None else X_num[part][idx]
        x_cat_batch = None if X_cat is None else X_cat[part][idx]
        # Build tokens using feature_tokenizer (handles num and cat)
        tokens = real_model.feature_tokenizer(x_num_batch, x_cat_batch)
        tokens = real_model.cls_token(tokens)
        for block in real_model.blocks:
            tokens = block(tokens)
        return real_model.head(tokens)

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in lib.IndexLoader(
                                    D.size(part), eval_batch_size, False, device
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                    stats['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)

    # %%
    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(lib.TRAIN, x), Y_device[lib.TRAIN][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                stats['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')