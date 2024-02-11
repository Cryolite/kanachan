FROM cryolite/kanachan

WORKDIR /workspace/data

ENTRYPOINT ["torchrun", "--nproc_per_node", "gpu", "--standalone", "-m", "kanachan.training.bert.phase3.train"]
