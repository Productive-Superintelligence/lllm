from typing import Dict, Any, List, Optional
import datetime as dt
from dataclasses import dataclass
import lllm.utils as U
from lllm.core.const import RCollections
import os

@dataclass
class Log:
    timestamp: str
    value: str
    metadata: Dict[str, Any]

class LogSession:
    def __init__(self, db, collection: str, session_name: str):
        self.db = db
        assert isinstance(collection, str), f"collection must be a string"
        self.collection = collection
        self.session_name = session_name

    def log(self, value: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a value with metadata, this should be a stable interface for all loggers
        """
        timestamp = dt.datetime.now().isoformat()
        payload = dict(metadata) if metadata else {}
        self.db.write(timestamp, value, payload, self.collection, self.session_name)

class LogCollection:
    def __init__(self, db, collection: str):
        self.db = db
        self.collection = collection

    def create_session(self, session_name: str) -> LogSession:
        return LogSession(self.db, self.collection, session_name)

class LogBase:
    """
    A base class for key-value based logging systems.
    """
    def __init__(self, base_name: str, config: Dict[str, Any]):
        self.config = config
        self.log_dir = U.pjoin(config['log_dir'], base_name)
        U.mkdirs(self.log_dir)
        
    def get_collection(self, collection: str) -> LogCollection:
        return LogCollection(self, collection)

    def write(self, key: str, value: str, metadata: Dict[str, Any], collection: str, session_name: str):
        raise NotImplementedError("write not implemented")

    def read(self, collection: str, session_name: str) -> List[Log]:
        raise NotImplementedError("read not implemented")

    def del_collection(self, collection: str):
        raise NotImplementedError("del_collection not implemented")

    def del_session(self, collection: str, session_name: str):
        raise NotImplementedError("del_session not implemented")
        
    def log_error(self, error: str):
        raise NotImplementedError("log_error not implemented")


class ReplayableLogBase(LogBase):

    def log_error(self, error: str):
        self.write(RCollections.ERRORS, error, {}, RCollections.ERRORS.value, self.session_name)

    def get_collection(self, collection: RCollections):
        assert collection in RCollections, f"collection must be one of RCollections: {RCollections}"
        return LogCollection(self, collection.value)


@dataclass
class Activity:
    timestamp: str
    value: str
    metadata: Dict[str, Any]
    type: str


class ReplaySession:
    def __init__(self, db, session_name: str):
        self.db = db
        self.session_name = session_name
        self.dialogs = self.db.read(RCollections.DIALOGS, self.session_name)
        self.frontend = self.db.read(RCollections.FRONTEND, self.session_name)
        self.messages = {}
        for dialog in self.dialogs:
            dialog_id = dialog.value
            self.messages[dialog_id] = self.db.read(RCollections.MESSAGES, f'{self.session_name}/{dialog_id}')

    @property
    def activities(self) -> List[Activity]: # inseart frontend messages to messages
        activities = []
        for dialog in self.dialogs:
            dialog_id = dialog.value
            activities.extend(self.messages[dialog_id])
        activities.extend(self.frontend)
        # sort by timestamp, oldest first
        activities.sort(key=lambda x: x.timestamp)
        return activities


class LocalFileLog(ReplayableLogBase):

    def write(self, key: str, value: str, metadata: Dict[str, Any], collection: str, session_name: str):
        folder = U.pjoin(self.log_dir, collection, session_name)
        U.mkdirs(folder)
        file_path = U.pjoin(folder, f"{key}.json")
        data = {
            'timestamp': key,
            'value': value,
            'metadata': metadata
        }
        U.save_json(file_path, data)

    def read(self, collection: str, session_name: str) -> List[Log]:
        folder = U.pjoin(self.log_dir, collection, session_name)
        if not os.path.exists(folder):
            return []
        files = os.listdir(folder)
        data = []
        for file in files:
            data.append(U.load_json(U.pjoin(folder, file)))
        # sort by timestamp, oldest first
        data.sort(key=lambda x: x['timestamp'])
        return [Log(**log) for log in data]

    def del_collection(self, collection: str):
        folder = U.pjoin(self.log_dir, collection)
        U.rmtree(folder)

    def del_session(self, collection: str, session_name: str):
        folder = U.pjoin(self.log_dir, collection, session_name)
        U.rmtree(folder)


class NoLog(ReplayableLogBase):
    def write(self, key: str, value: str, metadata: Dict[str, Any], collection: str, session_name: str):
        pass
    def read(self, collection: str, session_name: str) -> List[Log]:
        return []
    def del_collection(self, collection: str):
        pass
    def del_session(self, collection: str, session_name: str):
        pass
        
        
def build_log_base(config: Dict[str, Any], base_name: str = None):
    if base_name is None:
        base_name = config['name'] + '_default'
    if config['log_type'] == 'localfile':
        return LocalFileLog(base_name, config)
    elif config['log_type'] == 'none':
        return NoLog(base_name, config)
    else:
        raise ValueError(f"Log type {config['log_type']} not supported")
