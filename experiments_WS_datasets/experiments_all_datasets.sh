#### Conll Dataset
# Prediction
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset conll --batch_size 32 --lr 2e-5 --lr_crf 0.001 --lr_worker 0.001 --scaling 2.0 --run_times ${i};
  python neural-hidden-crf.py --dataset conll --batch_size 32 --lr 2e-5 --lr_crf 0.001 --lr_worker 0.001 --scaling 2.0 --run_times ${i};
done

# Inference
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset conll --batch_size 32 --lr 2e-5 --lr_crf 0.01 --lr_worker 0.001 --scaling 2.0 --run_on_testset True --run_times ${i};
  python neural-hidden-crf.py --dataset conll --batch_size 32 --lr 2e-5 --lr_crf 0.01 --lr_worker 0.001 --scaling 2.0 --run_on_testset True --run_times ${i};
done


#### Wikigold Dataset
# Prediction
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset wikigold --batch_size 16 --lr 2e-5 --lr_crf 0.005 --lr_worker 0.001 --scaling 2.0 --run_times ${i};
  python neural-hidden-crf.py --dataset wikigold --batch_size 16 --lr 2e-5 --lr_crf 0.005 --lr_worker 0.001 --scaling 2.0 --run_times ${i};
done

# Inference
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset wikigold --batch_size 32 --lr 3e-5 --lr_crf 0.001 --lr_worker 0.001 --scaling 2.0 --run_on_testset True --run_times ${i};
  python neural-hidden-crf.py --dataset wikigold --batch_size 32 --lr 3e-5 --lr_crf 0.001 --lr_worker 0.001 --scaling 2.0 --run_on_testset True --run_times ${i};
done


#### Mit-restaurants Dataset
# Prediction
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset mit-restaurants --batch_size 32 --lr 2e-5 --lr_crf 0.001 --lr_worker 0.01 --scaling 6.0 --run_times ${i};
  python neural-hidden-crf.py --dataset mit-restaurants --batch_size 32 --lr 2e-5 --lr_crf 0.001 --lr_worker 0.01 --scaling 6.0 --run_times ${i};
done

# Inference
for ((i=1;i<=10;i++)); do
  echo python neural-hidden-crf.py --dataset mit-restaurants --batch_size 16 --lr 2e-5 --lr_crf 0.01 --lr_worker 0.2 --scaling 5.0 --run_on_testset True --run_times ${i};
  python neural-hidden-crf.py --dataset mit-restaurants --batch_size 16 --lr 2e-5 --lr_crf 0.01 --lr_worker 0.2 --scaling 5.0 --run_on_testset True --run_times ${i};
done
