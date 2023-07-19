# mdx での Megatron-DeepSpeed のセットアップ方法

(2023/7/19 更新: 新 Megatron-DeepSpeed と旧 Megatron-DeepSpeed では基本のセットアップ方法にほぼ差はありません。)

## pyenv + venv でのセットアップ

基本的な手順は[この記事](https://zenn.dev/turing_motors/articles/04c1328bf6095a)と同様です。

1. pyenv が環境にあるかの確認

   ```bash
   > pyenv --version
   pyenv 2.3.21
   ```

   入っていない場合は `curl https://pyenv.run | bash`で install 可能です。

   mdx では、`~/.bashrc`が自動で読み込まれないようなので、pyenv を install した際は

   ```bash
   # pyenv
   export PYENV_ROOT="$HOME/.pyenv"
   command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init -)"

   eval "$(pyenv virtualenv-init -)"
   ```

   を `~/.bashrc` に追記すると思いますが、手動で `source ~/.bashrc` する必要があります。

2. pyenv で python を install

   ```bash
   > pyenv install 3.10.10
   > cd Megatron-DeepSpeed
   > pyenv local 3.10.10
   > python -m venv .env
   > source .env/bin/activate
   ```

   で、python の環境を作成します。

   この際、`pyenv local` で指定した python のバージョンが、`python --version` で表示されることを確認してください。
   (Python3.10.10, Python3.11.4 の両方で動作確認済みです。)

3. pip install

   `nvcc --version`で表示される cuda のバージョンに合わせて、`requirements.txt`を変更してください。

   Megatron-DeepSpeed を`git@github.com:llm-jp/Megatron-DeepSpeed.git`から clone してきた場合は、

   ```bash
   git switch hpc/fujii/deepspeed-multi-node
   pip install -r requirements.txt
   ```

   とすることで、CUDA11.8 に対応した PyTorch などが install されます。

4. apex install

   NVIDIA:apex を install します。

   ```bash
   git clone git@github.com:NVIDIA/apex.git
   cd apex
   ```

   ここで apex を install 際のコマンドを [こちら](https://github.com/NVIDIA/apex#linux)から確認ください。pip version によってコマンドが異なります。

   補足: apex の install がうまくいかない場合は、apex の commit を前に戻すと上手くいくことがあります。

5. `megatron/data/Makefile`を書き換える

   ```
   CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
   CPPFLAGS += $(shell python3 -m pybind11 --includes)
   LIBNAME = helpers
   LIBEXT = $(shell /home/kazukifujii/.pyenv/versions/3.11.4/bin/python3-config --extension-suffix)

   default: $(LIBNAME)$(LIBEXT)

   %$(LIBEXT): %.cpp
   	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
   ```

   のような形で、自分の環境に合わせた`python3-config`に書き換えてください。

   その後、`megatron/data`にて`make`を行うことを忘れないでください。

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

## mpirun

旧 Megatron-DeepSpeed の script は、`scripts/old_version`に移動しました。

`scripts/mpirun`に新 Megatron-DeepSpeed 対応の script があります。

標準出力と標準エラー出力を`outputs/log`に保存するようにしています。
また、`checkpoint`についても同様に`outputs/checkpoint`に保存するようにしています。

## deepspeed

旧 Megatron-DeepSpeed 対応のコードは `scripts/old_version/deepspeed`にあります。

また、新 Megatron-DeepSpeed 対応のコードは `scripts/deepspeed`にあります。

log, checkpoint の処理は mpirun と同様です。

### hostfile について

`scripts/deepspeed/hostfile`に hostfile を置くことを想定しています。
また hostfile は環境依存であり、git 管理するべき対象ではないため`.gitignore`に追加しています。

### マルチノード学習時の注意

pdsh の install が必要です。
すでに install されている node もありますが、すべてのノードに install されているとは限らないので注意してください。

PDSH の install について: https://zenn.dev/turing_motors/articles/dff1466194f4ac#%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97

## Appendix

pyenv を install する前に以下のようなコマンドを打ち

```bash
export PYENV_ROOT="model/<user-dir>/.pyenv"
```

bashrc に書き込むものも、上記のパスに合わせれば `~/`以下でないところに pyenv を install できます。

また python cache も

```bash
# pip cache
export PIP_CACHE_DIR="<user-dir>/.cache/pip"
```

とすることで、cache 作成先を変えることができます。
