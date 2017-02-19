device=$1
xp_dir=../log/$2

echo "Using device" $device
echo "Writing to directory: " $xp_dir

mkdir $xp_dir;
> $xp_dir/log_cifar100.txt;

# Baselines

# CIFAR-100 Adagrad pre-training
python baseline.py --dataset cifar100 --solver adagrad \
    --lr 0.01 --n_epochs 10 --loss ce --device $device --xp_dir $xp_dir;

# CIFAR-100 Adagrad training
python baseline.py --dataset cifar100 --solver adagrad \
    --lr 0.001 --n_epochs 2000 --save_at 1000 --loss svm \
    --device $device --xp_dir $xp_dir  --in_name cifar100_adagrad_ce;

# CIFAR-100 Adadelta pre-training
python baseline.py --dataset cifar100 --solver adadelta \
    --lr 1.0 --n_epochs 10 --loss ce --device $device --xp_dir $xp_dir;

# CIFAR-100 Adadelta training
python baseline.py --dataset cifar100 --solver adadelta \
    --lr 0.1 --n_epochs 200 --save_at 100 --loss svm \
    --device $device --xp_dir $xp_dir  --in_name cifar100_adadelta_ce;

# CIFAR-100 Adam pre-training
python baseline.py --dataset cifar100 --solver adam \
    --lr 0.001 --n_epochs 10 --loss ce --device $device --xp_dir $xp_dir;

# CIFAR-100 Adam training
python baseline.py --dataset cifar100 --solver adam \
    --lr 0.0001 --n_epochs 200 --save_at 100 --loss svm \
    --device $device --xp_dir $xp_dir  --in_name cifar100_adam_ce;

# LW-SVM runs

# CIFAR-100 Adagrad tuning
python lwsvm.py --dataset cifar100 --in_name cifar100_adagrad_svm --device $device --xp_dir $xp_dir;

# CIFAR-100 Adadelta tuning
python lwsvm.py --dataset cifar100 --in_name cifar100_adadelta_svm --device $device --xp_dir $xp_dir;

# CIFAR-100 Adam tuning
python lwsvm.py --dataset cifar100 --in_name cifar100_adam_svm --device $device --xp_dir $xp_dir;

# extract results
python extract.py --xp_dir $xp_dir --device $device;

# plot results and save
python visualize.py --xp_dir $xp_dir --export_pdf $xp_dir/plot.pdf;
