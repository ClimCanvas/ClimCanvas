# Server scripts (SSH tunnel + owner match)

[日本語](README.jp.md)

Helper scripts for running ClimCanvas on a shared Linux server as
**SSH tunnel + per-user process + owner match** (no reverse proxy): every user
runs their own ClimCanvas process under their own account on a private
loopback port, reaches it through an SSH tunnel, and an nftables owner-match
rule keeps other local users away from that port. The scripts combine port
assignment and the owner-match setup into one command.

The target is a Linux server with nftables. On a macOS development machine,
`CLIMCANVAS_DRYRUN=1` lets you inspect the generated rules without applying them.

## Files

- `climcanvas-ports.sh` — for the administrator (root). Maintains the port
  assignment table (`/etc/climcanvas/ports.tsv`) and generates and applies the
  nftables (owner-match) rules from it.
- `start-climcanvas.sh` — for users. Looks up the user's own port in the table
  and starts ClimCanvas on it (no need to remember the port number).

## Setup (administrator)

```bash
sudo install -m 0755 scripts/server/climcanvas-ports.sh /usr/local/bin/climcanvas-ports

# Assigning a port to a user = append to the table + apply the owner match, in one command
sudo climcanvas-ports assign alice               # pick a free port automatically
sudo climcanvas-ports assign bob --port 8502     # or choose the port
sudo climcanvas-ports list                       # list the assignments (UID, listening state)
sudo climcanvas-ports remove carol               # remove an assignment (rules are re-applied)
```

Every `assign` regenerates the nftables config from the whole table and
applies it, so rules never duplicate and stale assignments never linger. To
see the generated config before applying it, run `climcanvas-ports render`
(prints the nft config to stdout).

To make the nftables rules persistent, load the generated file
(`/etc/climcanvas/climcanvas.nft`) at boot, for example with an `include` in
the systemd nftables service or by adding
`include "/etc/climcanvas/climcanvas.nft"` to `/etc/nftables.conf`.

## Usage (users)

```bash
# 1. On your own machine, log in to the server with a tunnel to your assigned
#    port (here 8505; the administrator tells you, and it is shown by assign).
#    8501 is a port on your own machine: any free port will do.
ssh -L 8501:localhost:8505 alice@<server>

# 2. In that session on the server, start ClimCanvas under your own account
#    (the launcher looks up your port in the table)
scripts/server/start-climcanvas.sh
# To restrict the data directories the app may open:
CLIMCANVAS_ALLOWED_DIRS="$HOME/data:/shared/era5" scripts/server/start-climcanvas.sh

# 3. In a browser on your own machine, open http://localhost:8501
```

To keep ClimCanvas running permanently, wrap `start-climcanvas.sh` in a
systemd user service.

## What is protected (summary)

- **Authentication** = SSH login (limits who can reach the port at all)
- **Authorization** = per-user process + OS permissions (each user reads only
  what their account may read)
- **Closing the back door** (another local user connecting to port 8501
  directly) = owner match, which is what these scripts set up

The owner match expresses "only the owner and root may connect to
loopback:<port>; everyone else is rejected" in nftables. It protects the
loopback port only; it does not replace file permissions, and a user who can
run arbitrary code on the server can still read whatever their own account can read.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLIMCANVAS_PORTS_TABLE` | `/etc/climcanvas/ports.tsv` | path of the port table |
| `CLIMCANVAS_NFT_FILE` | `/etc/climcanvas/climcanvas.nft` | nft output file |
| `CLIMCANVAS_BASE_PORT` / `CLIMCANVAS_MAX_PORT` | `8501` / `8600` | range for automatic assignment |
| `CLIMCANVAS_DRYRUN` | `0` | `1` generates the rules without running nft (no root needed) |
| `CLIMCANVAS_APP_DIR` | `../..` from the script | location of `app.py` (launcher) |
| `CLIMCANVAS_PYTHON` | `python3` | python executable (launcher) |
| `CLIMCANVAS_ALLOWED_DIRS` | (none) | allowed data directories, passed on as `CC_ALLOWED_DIRS` (launcher) |
