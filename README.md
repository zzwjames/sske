## Dependencies
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

```python
# Cora
python main.py --der1 0.4 --der2 0.4 --dfr1 0.5 --dfr2 0.8 --temp 0.7
# Citeseer
python main_cite.py --dataname citeseer --der1 0.2 --der2 0.6 --dfr1 0.3 --dfr2 0.7 --temp 0.9 --epochs 200 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn prelu
# Pubmed
python main_pub.py --dataname pubmed --epochs 1500 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.4 --der2 0.1 --dfr1 0.0 --dfr2 0.6 --temp 1.0
# Photo
python main_photo.py --dataname photo --epochs 500 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.4 --der2 0.1 --dfr1 0.0 --dfr2 0.2 --temp 0.1
# Computer
python main_computer.py --dataname comp --epochs 500 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.2 --der2 0.1 --dfr1 0.0 --dfr2 0.2 --temp 0.1