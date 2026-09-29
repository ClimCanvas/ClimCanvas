# サーバ運用スクリプト (SSH トンネル + owner マッチ)

[English](README.en.md)

`docs/operate_server.md`「解決策 1-(a)」の運用 —
**SSH トンネル + per-user プロセス + owner マッチ**（リバースプロキシなし）を
補助するスクリプト群。ポート割り当てと owner マッチ設定を 1 コマンドにまとめる。

対象は Linux サーバ (nftables)。macOS の開発機では `CLIMCANVAS_DRYRUN=1` で
生成物の確認のみ可能。

## ファイル

- `climcanvas-ports.sh` — 管理者 (root) 用。ポート割り当て台帳
  (`/etc/climcanvas/ports.tsv`) の管理と、そこからの nftables (owner マッチ)
  ルール生成・適用。
- `start-climcanvas.sh` — 利用者用。台帳から自分のポートを引いて ClimCanvas を
  起動 (ポート番号を覚えなくてよい)。

スクリプトのコメントと実行時のメッセージは英語 (README だけ日本語版と英語版がある)。

## セットアップ (管理者)

```bash
sudo install -m 0755 scripts/server/climcanvas-ports.sh /usr/local/bin/climcanvas-ports

# ユーザにポートを割り当てる = 台帳追記 + owner マッチ適用が 1 コマンドで完了
sudo climcanvas-ports assign a-san          # 空きポートを自動割当
sudo climcanvas-ports assign b-san --port 8502   # ポート指定も可
sudo climcanvas-ports list                  # 割当一覧 (UID・待受状態)
sudo climcanvas-ports remove c-san          # 割当削除 (ルールも再適用)
```

`assign` するたびに台帳全体から nftables 設定を再生成して適用するので、
ルールが重複したり古い割当が残ったりしない。生成物を適用前に見るには
`climcanvas-ports render`（標準出力に nft 設定を表示）。

nftables ルールを永続化するには生成先 (`/etc/climcanvas/climcanvas.nft`) を
起動時に読み込む。例: systemd の nftables サービスの `include`、または
`/etc/nftables.conf` から `include "/etc/climcanvas/climcanvas.nft"`。

## 使い方 (利用者)

```bash
# 1. 手元の PC で、割り当てられたポート (ここでは 8505。管理者から通知され、
#    assign の案内にも出る) へのトンネルを張りつつサーバにログイン。
#    前半の 8501 は手元 PC 側のポートで、空いていれば何番でもよい
ssh -L 8501:localhost:8505 a-san@<server>

# 2. ログインしたサーバ上で起動 (自分の権限で。ポートは台帳から自動)
scripts/server/start-climcanvas.sh
# 許可データディレクトリを絞るなら:
CLIMCANVAS_ALLOWED_DIRS="$HOME/data:/shared/era5" scripts/server/start-climcanvas.sh

# 3. 手元 PC のブラウザで http://localhost:8501 を開く
```

常駐させたい場合は `start-climcanvas.sh` を systemd user service にする
(`docs/operate_server.md` D 節)。

## 何を守るか (要点)

- **認証** = SSH ログイン (ポートに届く人を制限)
- **認可** = per-user プロセス + OS パーミッション (読めるのは自分の権限内)
- **裏口 (C の 8501 直結) を塞ぐ** = owner マッチ ← このスクリプトが設定

owner マッチは「loopback:<port> に繋いでよいのは本人と root のみ、他は拒否」を
nftables で表現する。詳細と限界は `docs/operate_server.md`「解決策 1-(a)」。

## 環境変数

| 変数　 | 　既定　 |　 用途　 |
| --- | --- | --- |
| `CLIMCANVAS_PORTS_TABLE` | `/etc/climcanvas/ports.tsv` | 台帳のパス |
| `CLIMCANVAS_NFT_FILE` | `/etc/climcanvas/climcanvas.nft` | nft 出力先 |
| `CLIMCANVAS_BASE_PORT` / `CLIMCANVAS_MAX_PORT` | `8501` / `8600` | 自動割当の範囲 |
| `CLIMCANVAS_DRYRUN` | `0` | `1` で nft を実行せず生成のみ (root 不要) |
| `CLIMCANVAS_APP_DIR` | スクリプトの `../..` | `app.py` の場所 (launcher) |
| `CLIMCANVAS_PYTHON` | `python3` | python 実行ファイル (launcher) |
| `CLIMCANVAS_ALLOWED_DIRS` | (なし) | `CC_ALLOWED_DIRS` に渡す許可 dir (launcher) |

