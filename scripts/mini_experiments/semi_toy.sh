#SEMI=1 DETACH=0 TWOWAY=0 CE=0 python semi_toy.py
#SEMI=1 DETACH=0 TWOWAY=0 CE=1 python semi_toy.py
#SEMI=1 DETACH=0 TWOWAY=1 CE=0 python semi_toy.py
#SEMI=1 DETACH=0 TWOWAY=1 CE=1 python semi_toy.py
#SEMI=1 DETACH=1 TWOWAY=0 CE=0 python semi_toy.py
#SEMI=1 DETACH=1 TWOWAY=0 CE=1 python semi_toy.py
#SEMI=1 DETACH=1 TWOWAY=1 CE=0 python semi_toy.py
#SEMI=1 DETACH=1 TWOWAY=1 CE=1 python semi_toy.py

#python semi_toy.py --div_fn kl --seed 1
#python semi_toy.py --div_fn kl --seed 2
#python semi_toy.py --div_fn kl --seed 3
#python semi_toy.py --div_fn kl --seed 4
#python semi_toy.py --div_fn kl --seed 5
#python semi_toy.py --div_fn kl --seed 6
#python semi_toy.py --div_fn kl --seed 7
#python semi_toy.py --div_fn kl --seed 8
#python semi_toy.py --div_fn kl --seed 9
#python semi_toy.py --div_fn kl --seed 10
#python semi_toy.py --div_fn kl --seed 11
#python semi_toy.py --div_fn kl --seed 12
#python semi_toy.py --div_fn kl --seed 13
#python semi_toy.py --div_fn kl --seed 14
#python semi_toy.py --div_fn kl --seed 15
#python semi_toy.py --div_fn kl --seed 16
#python semi_toy.py --div_fn kl --seed 17
#python semi_toy.py --div_fn kl --seed 18
#python semi_toy.py --div_fn kl --seed 19
#python semi_toy.py --div_fn kl --seed 20
#python semi_toy.py --div_fn kl --seed 21
#python semi_toy.py --div_fn kl --seed 22
#python semi_toy.py --div_fn kl --seed 23
#python semi_toy.py --div_fn kl --seed 24
#python semi_toy.py --div_fn kl --seed 25
#python semi_toy.py --div_fn kl --seed 26
#python semi_toy.py --div_fn kl --seed 27
#python semi_toy.py --div_fn kl --seed 28
#python semi_toy.py --div_fn kl --seed 29
#python semi_toy.py --div_fn kl --seed 30
#python semi_toy.py --div_fn kl --seed 31
#python semi_toy.py --div_fn kl --seed 32
#python semi_toy.py --div_fn kl --seed 33
#python semi_toy.py --div_fn kl --seed 34
python semi_toy.py --div_fn kl --ylabel KL --xlabel "\$D(h(\bm x), h(\tilde{\bm x}))$"
#python semi_toy.py --div_fn rkl --ylabel rKL --xlabel "\$D(h(\tilde{\bm x}), p)$"
python semi_toy.py --detach_clean --div_fn kl --xlabel "\$D(\langle h(\bm x)\rangle, h(\tilde{\bm x}))$"
#python semi_toy.py --detach_clean --div_fn rkl  --xlabel "\$D(h(\tilde{\bm x}), \langle h(\bm x)\rangle)$"
python semi_toy.py --detach_pert --div_fn kl --xlabel "\$D(h(\bm x), \langle h(\tilde{\bm x})\rangle)$"
#python semi_toy.py --detach_pert --div_fn rkl --xlabel "\$D(\langle h(\tilde{\bm x})\rangle, h(\bm x))$"
python semi_toy.py --detach_pert --detach_clean --div_fn kl --xlabel "sup. only"
#python semi_toy.py --detach_pert --detach_clean --div_fn rkl --xlabel "sup. only 2"