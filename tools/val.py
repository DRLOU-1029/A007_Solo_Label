from typing import List

import torch
from tqdm import tqdm
def val_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics


def val_color_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            combined_inputs = torch.cat((inputs_l, inputs_r), dim=1)
            inputs = combined_inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics


def val_output_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs_l = model(inputs_l)
            outputs_r = model(inputs_r)
            outputs = outputs_l + outputs_r
            metric(outputs, labels)

    metrics = metric.compute_metric()

    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics


def val_stem_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs = model(inputs_l, inputs_r)
            metric(outputs, labels)

    metrics = metric.compute_metric()

    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics

"""def val_stem_merge_model(
        model,
        val_loader,
        metric,
        model_name="default_model",
        device='cuda'
):
    model.to(device)
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for inputs_l, inputs_r, labels in progress_bar:
            # 修改点：左右分支输入不再拼接，直接传入模型
            inputs_l = inputs_l.to(device)
            inputs_r = inputs_r.to(device)
            labels = labels.to(device)

            outputs = model(inputs_l, inputs_r)  # 直接传入两个分支输入
            metric(outputs, labels)  # 指标计算逻辑与原val一致

    metrics = metric.compute_metric()

    # 指标打印保持与原val相同
    for threshold in metric.thresholds:
        print(f'Threshold: {threshold}')
        print(f'Accuracy: {metrics[threshold]["accuracy"]:.4f}')
        print(f'Precision: {metrics[threshold]["precision"]:.4f}')
        print(f'Recall: {metrics[threshold]["recall"]:.4f}')

    return metrics  # 返回格式与原val一致"""