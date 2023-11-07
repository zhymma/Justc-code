import os

gpus = [0,0,2,2,2]

def run_folds():
    for i in range(5):  # Looping from fold0 to fold5
        command = f"nohup python run_train_mawps_fold_{i}.py --gpu_device {gpus[i]} > nohup_log_fold{i}.log 2>&1 &"
        os.system(command)

if __name__ == "__main__":
    run_folds()
