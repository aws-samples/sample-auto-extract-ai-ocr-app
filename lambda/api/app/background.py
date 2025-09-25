from threading import Thread
from queue import Queue
import json
import os
import requests
import logging
import traceback

logger = logging.getLogger(__name__)

class BackgroundTaskExtension(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.queue = Queue()
        self.session = requests.Session()
        self.start()
        logger.info("BackgroundTaskExtension initialized")

    def run(self):
        try:
            # Lambda環境かどうかを確認
            if 'AWS_LAMBDA_RUNTIME_API' in os.environ:
                # Lambda拡張APIに登録
                logger.info("Registering Lambda extension")
                response = self.session.post(
                    url=f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}/2020-01-01/extension/register",
                    json={'events': ['INVOKE'],},
                    headers={'Lambda-Extension-Name': 'background-task-extension'}
                )
                extension_id = response.headers['Lambda-Extension-Identifier']
                logger.info(f"Lambda extension registered with ID: {extension_id}")
                
                # イベントループを開始
                while True:
                    response = self.session.get(
                        url=f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}/2020-01-01/extension/event/next",
                        headers={'Lambda-Extension-Identifier': extension_id},
                        timeout=None
                    )
                    event = json.loads(response.text)
                    logger.info(f"Received Lambda event: {event['eventType']}")
                    
                    if event['eventType'] == 'INVOKE':
                        self._process_tasks()
            else:
                # 非Lambda環境（ローカル開発など）
                logger.info("Running in non-Lambda environment")
                while True:
                    self._process_tasks()
        except Exception as e:
            logger.error(f"Error in BackgroundTaskExtension: {str(e)}")
            logger.error(traceback.format_exc())

    def _process_tasks(self):
        """キューからタスクを処理"""
        while True:
            message = self.queue.get()
            if message['type'] == 'TASK':
                task_id = message.get('task_id', 'unknown')
                task_name = message['task'][0].__name__
                logger.info(f"Processing background task: {task_name} (ID: {task_id})")
                try:
                    task, args, kwargs = message['task']
                    task(*args, **kwargs)
                    logger.info(f"Background task completed: {task_name} (ID: {task_id})")
                except Exception as e:
                    logger.error(f"Error in background task {task_name} (ID: {task_id}): {str(e)}")
                    logger.error(traceback.format_exc())
            if message['type'] == 'DONE':
                logger.info("Received DONE signal, stopping task processing")
                break

    def add_task(self, background_task, *args, task_id=None, **kwargs):
        """タスクをキューに追加"""
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        logger.info(f"Adding task to queue: {background_task.__name__} (ID: {task_id})")
        self.queue.put({
            "type": "TASK",
            "task_id": task_id,
            "task": (background_task, args, kwargs)
        })
        return task_id

    def done(self):
        """現在のリクエストのタスク処理を完了"""
        logger.info("Marking current request as done")
        self.queue.put({"type": "DONE"})
