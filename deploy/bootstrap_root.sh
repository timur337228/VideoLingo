#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/srv/videolingo"
VENV_DIR="$PROJECT_DIR/.venv"

apt-get update
apt-get install -y \
  python3.12-venv python3-pip git ffmpeg fonts-noto \
  postgresql postgresql-contrib redis-server nginx certbot python3-certbot-nginx \
  build-essential libsndfile1

systemctl enable --now postgresql redis-server nginx

mkdir -p "$PROJECT_DIR"

if ! id -u www-data >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin www-data
fi

chown -R www-data:www-data "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install -U pip setuptools wheel
"$VENV_DIR/bin/pip" install gunicorn
"$VENV_DIR/bin/pip" install torch==2.8.0 torchaudio==2.8.0
"$VENV_DIR/bin/pip" install --no-deps "demucs[dev] @ git+https://github.com/adefossez/demucs"
"$VENV_DIR/bin/pip" install dora-search openunmix lameenc

cd "$PROJECT_DIR"
grep -vE '^(torch|torchaudio)==|^demucs @ git\+' requirements.txt > filtered-requirements.txt
"$VENV_DIR/bin/pip" install -r filtered-requirements.txt
rm -f filtered-requirements.txt

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/output" "$PROJECT_DIR/uploads"
chown -R www-data:www-data "$PROJECT_DIR"
