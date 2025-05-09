import torch 
import numpy as np
import csv
import os


class csv_saver():
    def __init__(self, path):
        self.path = path
        self.HEADER = ['Scene', '#frames', 'scale', 'shift', 'Delta1', 'Delta2', 'Delta3', 'SignedRelative', 'AbsoluteError', 'AbsoluteRelative', 'MeanSquaredError']
        self.initialised = False

    def save_metrics_csv(self, prediction, ground_truth, scale, shift, scene_name, valid_depth=None, frames='NotSaved'):
        
        if not self.initialised:
            if os.path.isfile(self.path):
                raise FileExistsError(f'csv File does already exist. Does not want to overwrite: {self.path}')
            else:
                with open(self.path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADER)
            self.initialised = True

        row = [scene_name, frames, scale, shift,
            1.-OutlierRatio(prediction, ground_truth, threshold=1.25, valid_depth=valid_depth),
            1.-OutlierRatio(prediction, ground_truth, threshold=1.25**2, valid_depth=valid_depth),
            1.-OutlierRatio(prediction, ground_truth, threshold=1.25**3, valid_depth=valid_depth),
            SignedRelativeDifference_Error(prediction, ground_truth, valid_depth=valid_depth),
            AbsoluteDifference_Error(prediction, ground_truth, valid_depth=valid_depth),
            AbsoluteRelativeDifference_Error(prediction, ground_truth, valid_depth=valid_depth),
            MeanSquared_Error(prediction, ground_truth, valid_depth=valid_depth)]
        
        with open(self.path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def summarize_metrics_csv(self, additional_infos_Header=None, additional_infos_data=None):
        data = {}
        with open(self.path, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    for key in row:
                        data[key] = []
                else:
                    for key in row:
                        if key in ['Scene']:
                            data[key].append(row[key])
                        else:
                            data[key].append(float(row[key]))
        
        overall_mean = []
        overall_var = []
        for key in self.HEADER:
            if key == 'Scene':
                overall_mean.append('Overall Mean')
                overall_var.append('Overall Variance')
            elif key == '#frames':
                if 'NotSaved' in data[key]:
                    overall_mean.append('--')
                    overall_var.append('--')
                else:
                    overall_mean.append(np.mean(data[key]))
                    overall_var.append(np.var(data[key]))
            else:
                overall_mean.append(np.mean(data[key]))
                overall_var.append(np.var(data[key]))
        
        
        with open(self.path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(overall_mean)
            writer.writerow(overall_var)
            if (additional_infos_Header is not None) & (additional_infos_data is not None):
                writer.writerow([])
                writer.writerow(additional_infos_Header)
                writer.writerow(additional_infos_data)


def check_backend(prediction, ground_truth):
    if type(prediction) != type(ground_truth):
        raise TypeError("prediction and ground_truth must be of the same type")

    backend = torch if isinstance(prediction, torch.Tensor) else np if isinstance(prediction, np.ndarray) else None
    if backend is None:
        raise TypeError("Unsupported type: expected torch.Tensor or np.ndarray")

    return backend

# Absolute Difference
def AbsoluteDifference(prediction, ground_truth, valid_depth=None):
    backend = check_backend(prediction, ground_truth)

    if valid_depth is not None:
        return backend.where(valid_depth, backend.absolute(prediction - ground_truth), 0.)
    else:
        return backend.absolute(prediction - ground_truth)

def AbsoluteDifference_Error(prediction, ground_truth, valid_depth=None):
    AbsDiff = AbsoluteDifference(prediction, ground_truth, valid_depth=valid_depth)
    
    if valid_depth is not None:
        return AbsDiff[valid_depth].mean()
    else:
        return AbsDiff.mean()

# Absolute Relative Difference
@np.errstate(divide='ignore')
def AbsoluteRelativeDifference(prediction, ground_truth, valid_depth=None):
    backend = check_backend(prediction, ground_truth)
    AbsDiff = AbsoluteDifference(prediction, ground_truth, valid_depth)
    
    if valid_depth is not None:
        return backend.where(valid_depth, AbsDiff / ground_truth, 0.)
    else:
        return  AbsDiff / ground_truth
    
def AbsoluteRelativeDifference_Error(prediction, ground_truth, valid_depth=None):
    AbsRelDiff = AbsoluteRelativeDifference(prediction, ground_truth, valid_depth=valid_depth)
    
    if valid_depth is not None:
        return AbsRelDiff[valid_depth].mean()
    else:
        return AbsRelDiff.mean()

# Signed Relative Difference
@np.errstate(divide='ignore')
def SignedRelativeDifference(prediction, ground_truth, valid_depth=None):
    backend = check_backend(prediction, ground_truth)

    if valid_depth is not None:
        return backend.where(valid_depth, (prediction - ground_truth) / ground_truth, 0.)
    else:
        return (prediction - ground_truth) / ground_truth, 0.
    
def SignedRelativeDifference_Error(prediction, ground_truth, valid_depth=None):
    SRelDiff = SignedRelativeDifference(prediction, ground_truth, valid_depth=valid_depth)
    if valid_depth is not None:
        return SRelDiff[valid_depth].mean()
    else:
        return SRelDiff.mean()

# Outlier ratio
@np.errstate(divide='ignore')
def Outlier(prediction, ground_truth, threshold=1.25, valid_depth=None):
    backend = check_backend(prediction, ground_truth)
    
    if valid_depth is not None:
        stack_1 = backend.where(valid_depth, prediction / ground_truth, 0.)
        stack_2 = backend.where(valid_depth, ground_truth / prediction, 0.)
    else: 
        stack_1 = prediction / ground_truth
        stack_2 = ground_truth / prediction

    if backend is torch:
        outlier = backend.where(backend.max(backend.stack([stack_1, stack_2], dim=0), dim=0) > threshold, 
                        backend.tensor(1.), 
                        backend.tensor(0.))
    else: 
        outlier = backend.where(backend.max(backend.stack([stack_1, stack_2], axis=0), axis=0) > threshold, 
                        1., 
                        0.)
    return outlier

def OutlierRatio(prediction, ground_truth, threshold=1.25, valid_depth=None):
    outlier = Outlier(prediction, ground_truth, threshold, valid_depth)

    if valid_depth is not None:
        outlier_ratio = outlier[valid_depth].mean()
    else:
        outlier_ratio = outlier.mean()

    return outlier_ratio

# Mean Squared Distance
def MeanSquared(prediction, ground_truth, valid_depth=None):
    backend = check_backend(prediction, ground_truth)
    
    if valid_depth is not None:
        return backend.where(valid_depth, (prediction - ground_truth) ** 2, 0.)
    else: 
        return (prediction - ground_truth) ** 2

def MeanSquared_Error(prediction, ground_truth, valid_depth=None):
    backend = check_backend(prediction, ground_truth)
    MeanSqu = MeanSquared(prediction, ground_truth, valid_depth=valid_depth)

    if valid_depth is not None:
        return backend.mean((prediction[valid_depth] - ground_truth[valid_depth]) ** 2)
    else:
        return backend.mean((prediction - ground_truth) ** 2)
 

