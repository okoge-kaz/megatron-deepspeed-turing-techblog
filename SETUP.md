# mdx での Megatron-DeepSpeed のセットアップ方法

## pyenv + venv でのセットアップ

基本的な手順は[この記事](https://zenn.dev/turing_motors/articles/04c1328bf6095a)と同様です。

1. pyenv が環境にあるかの確認
    ```bash
    > pyenv --version
    pyenv 2.3.21
    ```

    入っていない場合は `curl https://pyenv.run | bash`でinstall可能です。

    mdxでは、`~/.bashrc`が自動で読み込まれないようなので、pyenvをinstallした際は
    ```bash
    # pyenv
    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    eval "$(pyenv virtualenv-init -)"
    ```
    を `~/.bashrc` に追記すると思いますが、手動で `source ~/.bashrc` する必要があります。

    (ログインするたびに、`bashrc`を読み込むのは手間ですが、お願いします)

2. pyenv で python を install
    ```bash
    > pyenv install 3.10.10
    > cd Megatron-DeepSpeed
    > pyenv local 3.10.10
    > python -m venv .env
    > source .env/bin/activate
    ```
    で、pythonの環境を作成します。

    この際、`pyenv local` で指定したpythonのバージョンが、`python --version` で表示されることを確認してください。

3. pip install

    `nvcc --version`で表示されるcudaのバージョンに合わせて、`requirements.txt`を変更してください。

    Megatron-DeepSpeedを`git@github.com:llm-jp/Megatron-DeepSpeed.git`からcloneしてきた場合は、
    ```bash
    git switch hpc/fujii/deepspeed-multi-node
    pip install -r requirements.txt
    ```

    とすることで、CUDA11.8に対応したPyTorchなどがinstallされます。

4. apex install

    NVIDIA:apex を install します。
    ```bash
    git clone git@github.com:NVIDIA/apex.git
    cd apex
    ```

    ここで apex を install 際のコマンドを [こちら](https://github.com/NVIDIA/apex#linux)から確認ください。pip versionによってコマンドが異なります。

Setup 完了です。


## Multi-Node 学習のための準備

### ssh config

`~/.ssh/config` に使用する node に `ssh <name>` で接続できるように `config` を設定してください。

ユーザー名や秘密鍵名、node の IP アドレスなどは変更する必要がありますが、以下が参考になると思います。

```bash
Host mdx
  HostName llm-jp.zapto.org
  User kazukifujii
  ServerAliveInterval 15
  IdentityFile ~/.ssh/mdx

Host 10.2.72.135
  HostName 10.2.72.135
  User kazukifujii
  IdentityFile ~/.ssh/mdx
  ServerAliveInterval 15
  ProxyCommand ssh -W %h:%p mdx

Host 10.2.72.136
  HostName 10.2.72.136
  User kazukifujii
  IdentityFile ~/.ssh/mdx
  ServerAliveInterval 15
  ProxyCommand ssh -W %h:%p mdx
```

mdx に login した状態で `ssh <node-name>`で接続できることを確認してください。

### mpirun

[このファイル](https://github.com/llm-jp/Megatron-DeepSpeed/blob/hpc/fujii/deepspeed-multi-node/scripts/mpirun/345m_dp16.sh)を参考にしてください。

`-H `には、使用するノードの名前を記入してください。(自分は、`.ssh/config` の HostName と Host名が同じなので)

```bash
mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -H 10.2.72.135:8,10.2.72.136:8 \
  -x MASTER_ADDR=10.2.72.135 \
```

とします。MASTER_ADDR は、`-H` で指定したノードのうち、一つを指定してください。
基本的には、今ログインしているノードを指定すれば良いと思います。

上の場合では、`10.2.72.135`が今ログインしているnodeです。

`10.2.72.135`にて、以下のようにjobを投げます

```bash
bash scripts/mpirun/345m_dp16.sh
```

標準出力を保存したい場合は、`bash scripts/mpirun/345m_dp16.sh > log.txt` などとしてください。

## Appendix


pyenv を install する前に以下のようなコマンドを打ち

```bash
export PYENV_ROOT="model/<user-dir>/.pyenv"
```

bashrcに書き込むものも、上記のパスに合わせれば `~/`以下でないところにpyenvをinstallできます。

また python cache も

```bash
# pip cache
export PIP_CACHE_DIR="<user-dir>/.cache/pip"
```

とすることで、cache作成先を変えることができます。
