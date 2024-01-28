FROM cryolite/kanachan

WORKDIR /workspace/data

#ENV CUDA_LAUNCH_BLOCKING=1

ENTRYPOINT ["torchrun", "--nproc_per_node", "gpu", "--standalone", "-m", "kanachan.training.ranking.train"]
