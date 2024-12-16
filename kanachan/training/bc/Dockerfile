FROM cryolite/kanachan

WORKDIR /workspace/data

#ENV CUDA_LAUNCH_BLOCKING=1

ENTRYPOINT ["bash", "-c", "eval \"$(pyenv init -)\"; exec torchrun --nproc_per_node gpu --standalone -m kanachan.training.bc.train \"$@\"", "--"]
