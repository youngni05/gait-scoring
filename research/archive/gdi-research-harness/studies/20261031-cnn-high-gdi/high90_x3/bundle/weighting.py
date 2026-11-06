"""Only the current fit's training labels enter these fixed loss weights."""
import numpy as np

EDGES = np.array([60., 70., 80., 90., 100.])

def record_side_weights(y, mode):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y).all():
        raise ValueError('Expected finite training record x side labels')
    bins = np.digitize(y, EDGES)
    counts = np.bincount(bins.ravel(), minlength=6)
    if mode == 'baseline':
        weight = np.ones_like(y)
    elif mode == 'high90_x3':
        weight = np.where(y >= 90., 3., 1.)
    elif mode == 'rare_sqrt_cap4':
        factors = np.minimum(4., np.sqrt(counts.max() / np.maximum(counts, 1)))
        weight = factors[bins]
    else:
        raise ValueError('Unknown weighting mode: ' + mode)
    weight /= weight.mean()
    return weight, {'mode': mode, 'training_records': len(y), 'training_side_bin_counts': counts.tolist(),
                    'bin_edges': EDGES.tolist(), 'normalized_weight_min': float(weight.min()),
                    'normalized_weight_max': float(weight.max()),
                    'high90_total_loss_share': float(weight[y >= 90].sum()/weight.sum()),
                    'weights_source': 'current_fit_training_labels_only'}

def example_weights(x, y, mode):
    weight, audit = record_side_weights(y, mode)
    # Preserve the baseline's exact float32 values in baseline mode.
    result = (x.side_weights().reshape(-1, 2) * weight[x.owner]).astype(np.float32).reshape(-1)
    if not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError('Invalid example weights')
    return result, audit
