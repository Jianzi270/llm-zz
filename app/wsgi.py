"""生产 WSGI 入口。

Windows/内网部署示例：
  waitress-serve --listen=127.0.0.1:8000 app.wsgi:app
"""
from app.server import app

__all__ = ["app"]
