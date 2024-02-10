# Retreaval-based-Adversarial-Defense

    Adversarial defense by retreaval-based methods

----

Ideas of the retreaval-based methods:

- [ ] input pixel patch
- [ ] input pixel textual patch (f')
- [ ] cnn fmap patch
- [ ] cnn fmap textual patch


### quick start

⚪ Preparation

- download the datasets [here](https://pan.quark.cn/s/cb9b0dbd64f7), unzip to `data/` folder 
  - NIPS17 & ssa-cwa-200: clean and pre-generated adversarial images from [Attack-Bard](https://github.com/thu-ml/Attack-Bard)
  - imagenet-1k: 1000 cherry-picked images from the imagenet validation set

⚪ Warmup

- run `vis_NIPS17.py`, try understand what happens
- run `run_NIPS17_clf.py`, try understand what happens
- run `run.py`, try understand what happens
- run `run.py --atk`, try understand what happens

⚪ Your Tasks

> Use `imagenet-1k` as the ref-data to remove adv noise on `ssa-cwa-200` (pregen adv of `NIPS17`)
> Our final goal: let `run.py --atk --dfn` work! :)

- [x] implement `defenses.vector_db`
- [x] implement `defenses.img_hifreq`
- [ ] implement `defenses.patch_replace`


#### references

- rvc-project: [https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- the-ever-lasting-adversarial-war: [https://github.com/Kahsolt/the-ever-lasting-adversarial-war](https://github.com/Kahsolt/the-ever-lasting-adversarial-war)

----
by Armit
2023/10/26
