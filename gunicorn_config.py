import os

bind = f"0.0.0.0:{int(os.environ.get('PORT', 3000))}"
workers = int(os.environ.get('WEB_CONCURRENCY', 4))
timeout = 120
worker_class = 'sync'