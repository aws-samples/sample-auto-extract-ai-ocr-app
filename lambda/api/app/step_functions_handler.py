"""後方互換: Dockerfile.stepfunctions の CMD が参照

実体は workers/step_functions.py に移動済み。
Dockerfile.stepfunctions の CMD を app.workers.step_functions.process_image_handler に
変更した後、このファイルは削除可能。
"""
from app.workers.step_functions import process_image_handler  # noqa: F401
