#!/bin/bash

for i in {0..4}
do
    python "train_$i.py" --mode test --model_name "model_save_name_$i"
done
