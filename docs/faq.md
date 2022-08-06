---
layout: default
title: FAQ
nav_order: 8
---

# Frequently Asked Questions

{: .no_toc }

---
## Low performance compared to the original dataset
Since our data generation code is based on [pytorch-lightning](https://www.pytorchlightning.ai/), which supports a convenient toolkit for DDP(Distributed Data Parallel), several side-effects occur when you have installed a different version. For instance, in PyTorch-lightning 1.6.0, we found that the global step is fixed during the training phase. This affects on the learning rate since our code uses a learning rate scheduler based on the global step. 