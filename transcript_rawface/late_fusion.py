import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)

    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)

    return ccc, rho



# Fermin's 2018 tricks
def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    V = preds_flat
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0*(V-m1)/s1+m0
    return norm_preds



def get_Y(story, subject, smooth=0):
    file_name = "/Subject_"+str(subject)+"_Story_"+str(story) + ".csv"
    labels_path = "./data/original_dataset/annotations" + file_name  # Adjust this path to your actual labels directory
    Y = open(labels_path).read().split("\n")[1:-1]
    Y = [float(x) for x in Y]
    return Y


def get_all_Y(stories, subjects, normalize_labels=False, smooth=0):
    Y_list = []
    for subject in subjects:
        for story in stories:
            Y = get_Y(story, subject)
            Y_list.append(Y)
            if smooth>0:
                Y = butter_lowpass_filter_bidirectional(np.array(Y), cutoff=smooth, fs=25, order=1)
            if normalize_labels:
                Y = (Y- np.min(Y))/(np.max(Y)-np.min(Y))

    return np.concatenate(Y_list, axis=0)



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_bidirectional(data, cutoff=0.1, fs=25, order=1):
    y_first_pass = butter_lowpass_filter(data[::-1].flatten(), cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1].flatten(), cutoff, fs, order)
    return y_second_pass



# Removed hardcoded test_lenghts since we're computing dynamically from labels

# Evaluate with "average prediction" for each subject (WITHOUT filter optimization)
results = []
with_filter = True
subjects = [1,2,3,4,5,6,7,8,9,10]
modalities = ["rawface", "transcript"]  # Adapted for raw_face and transcript (lexicons)
stories_trainVal = [4,8]
stories_test = [5]
results_modality = {m:0 for m in modalities}

save_csv = True
save_path = 'test_prediction_FINAL/'  # Adjust this to your desired output path

finaldf = pd.DataFrame()
ccc_values = []  # NEW: Store CCC values for averaging
for i, subject in enumerate(subjects):
    for j, story in enumerate(stories_test):
        Y_trainVal_s = get_all_Y(stories_trainVal, [subject])


        X_coeff = {
                   "rawface":    .1,  # From original script
                   "transcript":  1. ,  # From original script (transcript/lexicons)
                  }

        filters = {
               "rawface":  (0.006,1),  # From original script
               "transcript":  (0.01,1),  # From original script
              }

        X = {}
        len_preds_s = len(get_Y(story, subject))  # Dynamically get the correct length from the labels
        preds_s = np.zeros((len_preds_s))
        ourdf = pd.DataFrame({"Subject":np.repeat(subject,len_preds_s)})
        for modality in modalities:
            file_name = "/Subject_"+str(subject)+"_Story_"+str(story)+"_predictions.npy"
            # base_path = "test/"  # Adjust this to the directory where your .npy predictions are stored
            # latent_vecs_path = base_path + modality + file_name
            latent_vecs_path = modality +"_predictions" + file_name
            X[modality] = np.load(latent_vecs_path)
            print(modality)
            print(X[modality].shape)
            X[modality] = X[modality].flatten()

            # NEW: Expand transcript predictions to match frame count by averaging overlaps
            if modality == "transcript":
                num_windows = len(X[modality])
                num_frames = len_preds_s
                full_preds = np.zeros(num_frames)
                counts = np.zeros(num_frames)
                window_size = 100  # Match from model_predictions.py
                stride = 50  # Match from model_predictions.py
                for win_i in range(num_windows):
                    start_i = win_i * stride
                    end_i = start_i + window_size
                    if end_i > num_frames:  # Safety trim (though unlikely)
                        end_i = num_frames
                    full_preds[start_i:end_i] += X[modality][win_i]
                    counts[start_i:end_i] += 1
                X[modality] = full_preds / (counts + 1e-10)  # Average where overlapped

            if X[modality].shape[0] != len_preds_s:
                print(f"Warning: Prediction length {X[modality].shape[0]} does not match label length {len_preds_s} for {modality}, Subject {subject}, Story {story}")
                # Optionally trim or pad, but for now, assume they match or handle accordingly
            X[modality] = butter_lowpass_filter_bidirectional(X[modality], cutoff=filters[modality][0], order=filters[modality][1])
            X[modality] = f_trick(Y_trainVal_s, X[modality])
            X[modality] = X[modality]*X_coeff[modality]
            preds_s += X[modality]
            ourdf[modality]=X[modality]
            finaldf = pd.concat([finaldf,ourdf])


        preds_s /= sum(X_coeff.values())



        if with_filter:
            preds_s = butter_lowpass_filter_bidirectional(preds_s, cutoff=0.01, order=1)
        preds_tricks_s = f_trick(Y_trainVal_s, preds_s)



        plt.figure(figsize=(13, 5))

        for modality in modalities:
            plt.plot(X[modality],label=modality)
        plt.plot(preds_tricks_s,label='average',lw=5)
        plt.legend()
        plt.show()

        # Compute and store CCC
        Y_test = np.array(get_Y(story, subject))
        ccc_val, rho = ccc(Y_test, preds_tricks_s)
        ccc_values.append(ccc_val)  # NEW: Store CCC value
        print(f"CCC for Subject {subject}, Story {story}: {ccc_val:.4f} (Pearson: {rho:.4f})")


        if save_csv:
            pd.DataFrame({"valence":preds_tricks_s}).to_csv(save_path+'Subject_{0}_Story_{1}.csv'.format(subject,story))
            pdddd = pd.DataFrame({"valence":preds_tricks_s})

# NEW: Compute and print average CCC
average_ccc = np.mean(ccc_values) if ccc_values else 0
print(f"Average CCC across all subjects and stories: {average_ccc:.4f}")