FROM continuumio/miniconda3:4.7.12

LABEL maintainer="Matthias Becker"

RUN conda config --add channels defaults
RUN conda config --add channels conda-forge
RUN conda config --add channels bioconda

RUN conda install -y snakemake=5.9.1 star=2.7.3a multiqc=1.8 samtools=1.9 ncurses=6.1 fastqc=0.11.9-0

ADD Snakefile /

ENV JOBS 32

CMD snakemake --snakefile /Snakefile --jobs $JOBS


FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

COPY ./sources.list /etc/apt/sources.list
COPY ./install_colabbatch_linux.sh /

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1397BC53640db551 \
&& apt update \
&& apt -y install gcc curl git wget gnutls-bin \
# && git config --global http.sslVerify false \
# && git config --global http.postBuffer 1048576000 \
&& git config --global user.name geng007 \
&& git config --global user.password 1997618hu \
&& bash /install_colabbatch_linux.sh

ENV PATH="/colabfold_batch/conda/bin:/colabfold_batch/conda/condabin:/colabfold_batch/bin:$PATH"
RUN echo "source activate /colabfold_batch/colabfold-conda" > ~/.bashrc

# CMD /bin/bash

##安装系统软件
# apt-get update -qq
# apt-get -qq -y install wget
# apt-get -qq -y install vim
##安装python依赖包
##5_TIANCHI
# ENV LANG C.UTF-8

# RUN python -m pip install --upgrade pip
# RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ pandas -y
# RUN pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install modlamp -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install catboost -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install xgboost --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install tape_proteins --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ##code_0.121
# # RUN pip install --upgrade setuptools && python -m pip install --upgrade pip
# # RUN python -m pip install --upgrade pip
# RUN pip install numpy 
# #RUN pip install pandas 
# #RUN pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install pandas --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install scikit-learn --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# #RUN pip install modlamp -i https://mirrors.ustc.edu.cn/pypi/web/simple
# RUN pip install cmake --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install tape_proteins --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install xgboost --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple

# ## jupyter notebook
# RUN pip install runipy --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install google.colab --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install jupyter --default-timeout=1688 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install "pyzmq==17.0.0" "ipykernel==4.8.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录
ADD . /
WORKDIR /
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
