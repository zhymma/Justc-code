import os

gpus = [6,6,6,5,5]

def run_folds():
    for i in range(1,5):  # Looping from fold0 to fold5
        command = f"nohup python run_train_mawps_fold_{i}.py --gpu_device {gpus[i]} > nohup_log_corrector{i}.log 2>&1 &"
        os.system(command)

if __name__ == "__main__":
    run_folds()
