python src/pipeline.py --dataset fb15k --model transe --negatives 100 --sampler relational --lr 0.001 --l2 0.00024036 --margin 1
python src/pipeline.py --dataset fb15k --model transe --negatives 40  --sampler relational --lr 0.001 --l2 0.00024036 --margin 1

python src/pipeline.py --dataset wn18 --model transe --negatives 100 --sampler relational --lr 0.01 --l2 0.00024036 --margin 1
python src/pipeline.py --dataset wn18 --model transe --negatives 40  --sampler relational --lr 0.01 --l2 0.00018637 --margin 1