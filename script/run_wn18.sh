
python src/pipeline.py --dataset wn18 --model transh --negatives 10 --sampler relational --lr 0.001 --l2 0.00024036 --margin 1
python src/pipeline.py --dataset wn18 --model transr --negatives 10 --sampler relational --lr 0.001 --l2 0.00024036 --margin 1
python src/pipeline.py --dataset wn18 --model transe --negatives 10 --sampler relational --lr 0.001 --l2 0.00024036 --margin 1

