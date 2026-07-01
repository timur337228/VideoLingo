# Production Deploy Files

This directory contains the production templates prepared for the current
project layout:

- `bootstrap_root.sh` installs system packages and Python dependencies.
- `systemd/` contains service units for Django, FastAPI and Celery.
- `nginx/` contains the Nginx virtual host config for `mixxtranslate.ru`.

These files assume the project is deployed to `/srv/mixxtranslate/app`.
