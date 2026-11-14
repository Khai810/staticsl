"""Metrics mức ảnh độc lập với detection mAP và positive nothing."""
from __future__ import annotations
from collections import Counter
import math

NO_DETECTION = 'NO_DETECTION'


def wilson(correct: int, total: int) -> list[float] | None:
    """Khoảng Wilson 95%, N/A khi mẫu số bằng 0."""
    if not total:
        return None
    z = 1.959963984540054
    p, d = correct / total, 1 + z*z/total
    c = (p + z*z/(2*total)) / d
    h = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [max(0., c-h), min(1., c+h)]


def top_detection(detections: list[dict], threshold: float) -> dict | None:
    """Chọn confidence cao nhất; tie giữ thứ tự sau NMS."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid confidence threshold')
    for d in detections:
        if not math.isfinite(d['confidence']) or not 0 <= d['confidence'] <= 1:
            raise ValueError('Invalid prediction confidence')
    accepted = [d for d in detections if d['confidence'] >= threshold]
    return max(accepted, key=lambda d: d['confidence']) if accepted else None


def score_predictions(samples: list[dict], predictions: dict, classes: list[str], supported: list[str], threshold: float) -> dict:
    """Chấm mọi positive, kể cả rejection; background báo riêng."""
    if set(predictions) != {s['sample_id'] for s in samples}:
        raise ValueError('Predictions must contain every sample exactly once')
    actual_supported = {s['label'] for s in samples if s['boxes']}
    if set(supported) != actual_supported or len(supported) != len(set(supported)):
        raise ValueError('Supported classes must equal positive validation classes')
    cm = [[0] * (len(classes)+1) for _ in classes]
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    backgrounds = bg_rejections = accepted = correct = total = multiple = 0
    records, pairs = [], Counter()
    for sample in samples:
        detections = predictions[sample['sample_id']]
        if any(type(d['class_id']) is not int or not 0 <= d['class_id'] < len(classes) for d in detections):
            raise ValueError('Predicted class ID outside mapping')
        top = top_detection(detections, threshold)
        pred = classes[top['class_id']] if top else NO_DETECTION
        multiple += sum(d['confidence'] >= threshold for d in detections) > 1
        bg = not sample['boxes']
        if bg:
            backgrounds += 1
            bg_rejections += top is None
        else:
            total += 1
            truth = sample['label']
            support[truth] += 1
            accepted += top is not None
            correct += pred == truth
            cm[sample['class_id']][top['class_id'] if top else len(classes)] += 1
            if pred == truth:
                tp[truth] += 1
            else:
                fn[truth] += 1
                if top:
                    fp[pred] += 1
                pairs[(truth, pred)] += 1
        records.append(dict(sample_id=sample['sample_id'], truth='BACKGROUND' if bg else sample['label'],
                            predicted=pred, confidence=top['confidence'] if top else None,
                            correct=(top is None) if bg else pred == sample['label'], background=bg))
    if not total:
        raise ValueError('Need positive validation images')
    per_class = {}
    for c in classes:
        p = tp[c] / (tp[c]+fp[c]) if tp[c]+fp[c] else 0.
        r = tp[c] / support[c] if support[c] else None
        f1 = 2*tp[c] / (2*tp[c]+fp[c]+fn[c]) if support[c] else None
        per_class[c] = dict(precision=p if support[c] or fp[c] else None, recall=r, f1=f1,
                            support=support[c], tp=tp[c], fp=fp[c], fn=fn[c],
                            background_fp=sum(x['background'] and x['predicted'] == c for x in records))
    average = lambda k: sum(per_class[c][k] for c in supported) / len(supported)
    return dict(threshold=threshold, accuracy=correct/total, correct=correct, n_positive=total,
                accuracy_wilson95=wilson(correct, total), val_macro_f1_supported=average('f1'),
                macro_precision_supported=average('precision'), macro_recall_supported=average('recall'),
                weighted_f1=sum(per_class[c]['f1']*support[c] for c in supported)/total,
                supported_classes=supported, macro_class_count=len(supported),
                excluded_from_macro=[c for c in classes if c not in supported], per_class=per_class,
                confusion_matrix=cm, confusion_rows=classes, confusion_columns=classes+[NO_DETECTION],
                accepted_positive=accepted, coverage=accepted/total,
                accepted_accuracy=correct/accepted if accepted else None,
                no_detection_positive=total-accepted, multiple_detection_images=multiple,
                n_background=backgrounds, background_correct_rejections=bg_rejections,
                background_correct_rejection_rate=bg_rejections/backgrounds if backgrounds else None,
                background_false_positives=backgrounds-bg_rejections,
                background_false_positive_rate=(backgrounds-bg_rejections)/backgrounds if backgrounds else None,
                top_confusions=[dict(truth=a,predicted=b,count=n) for (a,b),n in pairs.most_common(20)],
                mn_confusions={f'{a}_to_{b}':pairs[(a,b)] for a,b in [('M','N'),('N','M')]},
                records=records, split='val', average_scope='positive images only; background FP reported separately')


def is_better(candidate: dict, previous: dict | None) -> bool:
    """Ưu tiên macro F1, Accuracy, rồi epoch sớm hơn."""
    def key(x: dict) -> tuple:
        return (x['val_macro_f1_supported'], x['accuracy'], -x['epoch'])
    return previous is None or key(candidate) > key(previous)
