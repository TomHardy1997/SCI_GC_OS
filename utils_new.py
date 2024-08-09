import torch
import numpy as np
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored

def evaluate_on_test(model, test_loader, device):
    model.eval()
    test_all_risk_scores = []
    test_all_censorships = []
    test_all_event_times = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test Evaluation"):
            test_path_features, test_label, test_sur_time, test_censor, _ = data
            
            test_path_features = test_path_features.to(device)
            test_label = test_label.to(device)
            test_sur_time = test_sur_time.to(device)
            test_censor = test_censor.to(device)

            with torch.cuda.amp.autocast():
                test_outputs = model(test_path_features)
                test_hazards = torch.sigmoid(test_outputs)
                test_survival = torch.cumprod(1 - test_hazards, dim=1)
                test_risk = -torch.sum(test_survival, dim=1).detach().cpu().numpy()

            test_all_risk_scores.append(test_risk)
            test_all_censorships.append(test_censor.cpu().numpy().tolist())
            test_all_event_times.append(test_sur_time.cpu().numpy().tolist())

    test_all_risk_scores = np.concatenate(test_all_risk_scores)
    test_all_censorships = np.concatenate(test_all_censorships)
    test_all_event_times = np.concatenate(test_all_event_times)
    test_c_index = concordance_index_censored((1 - test_all_censorships).astype(bool), test_all_event_times, test_all_risk_scores, tied_tol=1e-08)[0]
    return test_c_index
