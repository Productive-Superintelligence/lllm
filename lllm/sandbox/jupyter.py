import os
import uuid
import subprocess
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import nbformat # For interacting with .ipynb files
from jupyter_client.manager import KernelManager # For starting and managing a kernel
from jupyter_client.blocking import BlockingKernelClient # For communicating with the kernel
import queue # For non-blocking message retrieval
import datetime as dt
import requests
from enum import Enum
import lllm.utils as U

logger = logging.getLogger(__name__)


class JupyterCellType(str, Enum):
    MARKDOWN = 'markdown'
    CODE = 'code'

class ProgrammingLanguage(str, Enum):
    PYTHON = 'python' # now only support python
    # R = 'r' # TODO: maybe add support for R


@dataclass
class JupyterSession:
    """Represents a Jupyter Notebook Session, including its server process and kernel."""
    name: str
    dir: str # Path to the directory containing the notebook file
    metadata: Dict[str, Any]
    notebook_file: Optional[str] = None
    programming_language: ProgrammingLanguage = ProgrammingLanguage.PYTHON

    # For Jupyter Server process (launching the web UI)
    server_process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    server_url: Optional[str] = field(default=None, init=False, repr=False)
    server_port: Optional[int] = field(default=None, init=False, repr=False)

    # For direct Kernel interaction (programmatic cell execution)
    kernel_manager: Optional[KernelManager] = field(default=None, init=False, repr=False)
    kernel_client: Optional[BlockingKernelClient] = field(default=None, init=False, repr=False)
    last_stop_index: int = 0 # if error, reset it

    def __post_init__(self):
        self.init_session()
        if self.metadata.get('autorun', False):
            self.run_all_cells()

    def init_session(self):
        _cutoff_date = self.metadata['proxy']['cutoff_date']
        if _cutoff_date:
            if isinstance(_cutoff_date, str):
                _cutoff_date = dt.datetime.strptime(_cutoff_date, '%Y-%m-%d')
            assert isinstance(_cutoff_date, dt.datetime), "Cutoff date must be a datetime object"
            _cutoff_date_str = f"'{_cutoff_date.strftime('%Y-%m-%d')}'"
        else:
            _cutoff_date_str = 'None'
        _project_root = self.metadata['project_root']
        if os.name == 'nt':
            _project_root = _project_root.replace('\\', '/')
        _init_code = f'''# INIT CODE (DO NOT REMOVE THIS CELL)
import sys
sys.path.append('{_project_root}')
from lllm.proxies import ProxyManager
proxy = ProxyManager(activate_proxies={self.metadata['proxy']['activate_proxies']}, cutoff_date={_cutoff_date_str}, deploy_mode={self.metadata['proxy']['deploy_mode']})
CALL_API = proxy.__call__'''
        cell_0_content = self.cells[0].source if self.cells else ''
        if not cell_0_content.strip().startswith('# INIT CODE'):    
            self.insert_cell(0, _init_code, JupyterCellType.CODE)
        else:
            self.overwrite_cell(0, _init_code, JupyterCellType.CODE)

    def to_dict(self) -> Dict[str, Any]:
        _metadata = self.metadata.copy()
        _datetime = _metadata['proxy']['cutoff_date']
        if isinstance(_datetime, dt.datetime):
            _metadata['proxy']['cutoff_date'] = _datetime.strftime('%Y-%m-%d')
        elif isinstance(_datetime, str):
            _metadata['proxy']['cutoff_date'] = _datetime
        else:
            _metadata['proxy']['cutoff_date'] = None
        return {
            'name': self.name,
            'dir': self.dir,
            'metadata': _metadata,
            'notebook_file': self.notebook_file,
            'programming_language': self.programming_language.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JupyterSession':
        # Ensure notebook_file is absolute or resolved correctly if needed
        session_dir = data.get('dir')
        notebook_file = data.get('notebook_file')
        if session_dir and notebook_file and not os.path.isabs(notebook_file):
            notebook_file = U.pjoin(session_dir, os.path.basename(notebook_file))

        _metadata = data['metadata'].copy()
        _datetime = _metadata['proxy']['cutoff_date']
        if isinstance(_datetime, str):
            _metadata['proxy']['cutoff_date'] = dt.datetime.strptime(_datetime, '%Y-%m-%d')
        elif isinstance(_datetime, dt.datetime):
            _metadata['proxy']['cutoff_date'] = _datetime
        else:
            _metadata['proxy']['cutoff_date'] = None
        return cls(
            name=data['name'],
            dir=session_dir,
            metadata=_metadata,
            notebook_file=notebook_file,
            programming_language=ProgrammingLanguage(data['programming_language']),
        )


    def _read_notebook_object(self) -> Optional[nbformat.NotebookNode]:
        if not self.notebook_file or not U.pexists(self.notebook_file):
            return nbformat.v4.new_notebook()
        try:
            with open(self.notebook_file, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except Exception as e:
            logger.warning("Error reading notebook %s: %s", self.notebook_file, e)
            return nbformat.v4.new_notebook()

    def _read_notebook_cells(self) -> List[nbformat.NotebookNode]:
        nb = self._read_notebook_object()
        return nb.cells if nb else []

    @property
    def cells(self) -> List[nbformat.NotebookNode]:
        return self._read_notebook_cells()
    
    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def get_cells(self, index: int | List[int]) -> List[nbformat.NotebookNode]:
        if isinstance(index, int):
            index = [index]
        return [self.cells[i] for i in index]

    @property
    def directory_tree(self) -> str:
        return U.directory_tree(self.dir)

    def _ensure_notebook_file(self, create: bool = True) -> None:
        if not self.notebook_file:
            self.notebook_file = U.pjoin(self.dir, f"{self.name}.ipynb")
        if create and not U.pexists(self.notebook_file):
            nb = nbformat.v4.new_notebook()
            try:
                with open(self.notebook_file, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                logger.debug("Created empty notebook: %s", self.notebook_file)
            except Exception as e:
                logger.error("Error creating notebook file %s: %s", self.notebook_file, e)
                self.notebook_file = None

    def _write_notebook_object(self, nb: nbformat.NotebookNode):
        if not self.notebook_file:
            logger.error("Notebook file path is not set, cannot write")
            return
        try:
            with open(self.notebook_file, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        except Exception as e:
            logger.error("Error writing to notebook file %s: %s", self.notebook_file, e)
            
    def _write_cell(self, content: str, cell_type: JupyterCellType,
                    ensure_exists: bool = True,
                    overwrite_index: Optional[int] = None,
                    insert_index: Optional[int] = None) -> int: # return the index of the cell
        self._ensure_notebook_file(create=ensure_exists)
        if not self.notebook_file:
            raise ValueError("Error: Cannot write cell, notebook file not set or couldn't be created.")
        
        assert overwrite_index is None or insert_index is None, "Cannot specify both overwrite_index and insert_index"

        nb = self._read_notebook_object()
        if not nb: 
            nb = nbformat.v4.new_notebook() # Should be handled by _read_notebook_object but defensive

        if cell_type == JupyterCellType.MARKDOWN:
            new_cell = nbformat.v4.new_markdown_cell(content)
        elif cell_type == JupyterCellType.CODE:
            new_cell = nbformat.v4.new_code_cell(content)
        else:
            raise ValueError(f"Invalid cell type: {cell_type}")
    
        if overwrite_index is not None:
            if 0 <= overwrite_index < len(nb.cells):
                nb.cells[overwrite_index] = new_cell
            else:
                raise ValueError(f"Error: overwrite_index {overwrite_index} out of bounds for {len(nb.cells)} cells.")
        elif insert_index is not None:
            if 0 <= insert_index <= len(nb.cells):
                nb.cells.insert(insert_index, new_cell)
            else:
                raise ValueError(f"Error: insert_index {insert_index} out of bounds for {len(nb.cells)} cells.")
        else:
            nb.cells.append(new_cell)

        self._write_notebook_object(nb)
        logger.debug("Modified %s cell in %s", cell_type.value, self.notebook_file)
        return len(nb.cells) - 1
    
    def append_code_cell(self, content: str, ensure_exists: bool = True) -> int:
        return self._write_cell(content, JupyterCellType.CODE, ensure_exists)
    
    def append_markdown_cell(self, content: str, ensure_exists: bool = True) -> int:
        return self._write_cell(content, JupyterCellType.MARKDOWN, ensure_exists)

    def overwrite_cell(self, index: int, content: str, cell_type: JupyterCellType):
        self._write_cell(content, cell_type, ensure_exists=True, overwrite_index=index)

    def insert_cell(self, index: int, content: str, cell_type: JupyterCellType):
        self._write_cell(content, cell_type, ensure_exists=True, insert_index=index)

    def delete_cells(self, index: int | List[int]):
        if isinstance(index, int):
            index = [index]
        to_delete = []
        nb = self._read_notebook_object()
        for i in index:
            if not nb or not (0 <= i < len(nb.cells)):
                raise ValueError(f"Error: Cannot delete cell at index {i}. Notebook/cell not found or index out of bounds.")
            to_delete.append(nb.cells[i])
        for cell in to_delete:
            nb.cells.remove(cell)
        self._write_notebook_object(nb)
        logger.debug("Deleted cells at indices %s from %s", index, self.notebook_file)


    # --- Jupyter Server (Web UI) Methods ---
    def launch_server(self, specific_port: Optional[int] = None) -> Optional[str]:
        if self.server_process and self.server_process.poll() is None:
            logger.debug("Server for session '%s' already running at %s", self.name, self.server_url)
            return self.server_url

        self._ensure_notebook_file(create=True)
        command = ['jupyter', 'notebook', f'--notebook-dir={self.dir}', '--no-browser', '--ip=127.0.0.1']
        if specific_port is not None:
            command.append(f'--port={specific_port}')
        else:
            command.append('--port-retries=50')

        logger.debug("Launching Jupyter server for session '%s' in '%s'", self.name, self.dir)
        try:
            self.server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
            url_pattern = re.compile(r'(http://(127\.0\.0\.1|localhost):(\d+)/\?token=\w+)')
            start_time = time.time()
            timeout_seconds = 20
            
            stderr_output = ""
            while time.time() - start_time < timeout_seconds:
                if self.server_process.stderr:
                    line = self.server_process.stderr.readline()
                    if line:
                        stderr_output += line
                        # print(f"[JupyterServer stderr] {line.strip()}") # Can be very verbose
                        match = url_pattern.search(line)
                        if match:
                            self.server_url = match.group(1)
                            if self.notebook_file and U.pexists(self.notebook_file):
                                nb_filename = os.path.basename(self.notebook_file)
                                encoded_nb_filename = requests.utils.quote(nb_filename)
                                self.server_url = f"{self.server_url.split('?')[0]}notebooks/{encoded_nb_filename}?{self.server_url.split('?')[1]}"
                            
                            port_match = re.search(r':(\d+)/', self.server_url)
                            if port_match: self.server_port = int(port_match.group(1))
                            logger.info("Jupyter server started: %s (PID %s)", self.server_url, self.server_process.pid)
                            return self.server_url
                    elif self.server_process.poll() is not None:
                        logger.error("Jupyter server process terminated unexpectedly (exit code %s)", self.server_process.poll())
                        break
                    time.sleep(0.1)
                else: time.sleep(0.2)
            
            logger.error("Could not find Jupyter server URL in stderr within %ds", timeout_seconds)
            self.shutdown_server()
            return None
        except FileNotFoundError:
            logger.error("'jupyter' command not found for server")
            self.server_process = None
            return None
        except Exception as e:
            logger.error("Unexpected error during Jupyter server launch: %s", e, exc_info=True)
            if self.server_process: self.shutdown_server()
            return None

    def shutdown_server(self):
        if self.server_process and self.server_process.poll() is None:
            logger.info("Shutting down Jupyter server for session '%s' (PID %s)", self.name, self.server_process.pid)
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill(); self.server_process.wait()
            finally:
                if self.server_process.stdout: self.server_process.stdout.close()
                if self.server_process.stderr: self.server_process.stderr.close()
            logger.debug("Jupyter server shut down")
        self.server_process, self.server_url, self.server_port = None, None, None

    # --- Kernel Interaction Methods ---
    def start_kernel(self) -> bool:
        if self.kernel_manager and self.kernel_manager.is_alive():
            # print("Kernel already running for this session.")
            return True
        

        lock = U.make_file_lock('lllm_jupyter_kernel', timeout=20)

        logger.debug("Starting kernel for session '%s'", self.name)
        try:
            with lock:
                self.kernel_manager = KernelManager(kernel_name='python3', env=os.environ)
                self.kernel_manager.start_kernel()
                self.kernel_client = self.kernel_manager.client()
                self.kernel_client.start_channels()
                try:
                    self.kernel_client.wait_for_ready(timeout=10)
                    logger.debug("Kernel started and ready (ID: %s)", self.kernel_manager.kernel_id)
                    self.last_stop_index = 0
                    return True
                except RuntimeError:
                    logger.error("Timeout waiting for kernel to become ready")
                    self.shutdown_kernel()
                    return False
        except TimeoutError:
            logger.error("Failed to acquire lock on lllm_jupyter_kernel")
            self.shutdown_kernel()
            return False
        except Exception as e:
            logger.error("Failed to start kernel: %s", e, exc_info=True)
            self.shutdown_kernel()
            return False

    def shutdown_kernel(self):
        client_stopped, manager_stopped = False, False
        if self.kernel_client:
            try: self.kernel_client.stop_channels(); client_stopped = True
            except Exception as e:
                logger.warning("Error stopping client channels: %s", e)
        if self.kernel_manager and self.kernel_manager.is_alive():
            try: self.kernel_manager.shutdown_kernel(now=True); manager_stopped = True
            except Exception as e:
                logger.warning("Error shutting down kernel: %s", e)
        elif self.kernel_manager:
            manager_stopped = True

        self.kernel_client, self.kernel_manager = None, None
        if client_stopped or manager_stopped:
            logger.debug("Kernel resources released")
        self.last_stop_index = 0

    def run_cell(self, index: int, timeout: int = 60) -> bool:
        if not self.start_kernel():
            logger.error("Cannot run cell %d: kernel failed to start", index)
            return False

        nb = self._read_notebook_object()
        if not nb or not (0 <= index < len(nb.cells)):
            logger.error("Cell index %d out of bounds or notebook not found", index)
            return False
        
        cell_to_run = nb.cells[index]
        if cell_to_run.cell_type != 'code':
            return True 

        code = cell_to_run.source
        if not code.strip():
            cell_to_run.outputs = [] 
            self._write_notebook_object(nb)
            return True

        # This print is already in user's logs:
        # print(f"\nExecuting cell {index} in kernel for session '{self.name}':\n---\n{code}\n---")
        msg_id = self.kernel_client.execute(code, store_history=True)
        cell_to_run.outputs = [] 
        
        # Loop to gather IOPub messages
        stop_iopub_gathering_loop = False
        iopub_loop_start_time = time.monotonic()
        max_iopub_loop_duration = timeout - 2 


        while not stop_iopub_gathering_loop:
            current_loop_duration = time.monotonic() - iopub_loop_start_time
            if current_loop_duration > max_iopub_loop_duration and max_iopub_loop_duration > 0 :
                break

            try:
                # Poll iopub with a short timeout to remain responsive
                msg = self.kernel_client.get_iopub_msg(timeout=0.2) 
                
                if msg['parent_header'].get('msg_id') == msg_id:
                    try:
                        output = nbformat.v4.output_from_msg(msg)
                        cell_to_run.outputs.append(output)
                    except ValueError: 
                        pass 
                    
                    if msg['header']['msg_type'] == 'status' and \
                       msg['content']['execution_state'] == 'idle':
                        stop_iopub_gathering_loop = True
            
            except queue.Empty:
                try:
                    shell_check = self.kernel_client.get_shell_msg(timeout=0) # Non-blocking check
                    if shell_check and shell_check['parent_header'].get('msg_id') == msg_id:
                        stop_iopub_gathering_loop = True # Shell reply means execution is done
                except queue.Empty:
                    pass 
            
            except Exception as e:
                logger.warning("Error processing iopub message for cell %d: %s", index, e)
                stop_iopub_gathering_loop = True

        execution_successful = False
        remaining_timeout_for_shell = max(1, timeout - (time.monotonic() - iopub_loop_start_time))

        try:
            shell_reply = self.kernel_client.get_shell_msg(timeout=remaining_timeout_for_shell)
            if shell_reply['parent_header'].get('msg_id') == msg_id:
                status = shell_reply['content']['status']
                logger.debug("Kernel execution status for cell %d: %s", index, status)
                if status == 'ok':
                    cell_to_run.execution_count = shell_reply['content'].get('execution_count')
                    execution_successful = True
                elif status == 'error':
                    if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                        err_output = nbformat.v4.new_output(
                            output_type='error',
                            ename=shell_reply['content']['ename'],
                            evalue=shell_reply['content']['evalue'],
                            traceback=shell_reply['content']['traceback']
                        )
                        cell_to_run.outputs.append(err_output)
            else:
                logger.warning("Mismatched shell reply for cell %d", index)
                if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                    cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='ShellReplyError', evalue='Mismatched shell reply ID', traceback=[]))
        
        except queue.Empty:
            logger.warning("Timeout (%.1fs) waiting for shell reply for cell %d", remaining_timeout_for_shell, index)
            if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='TimeoutError', evalue='Timeout waiting for shell reply', traceback=[]))
        except Exception as e:
            logger.error("Error getting shell reply for cell %d: %s", index, e)
            if not any(out.output_type == 'error' for out in cell_to_run.outputs):
                cell_to_run.outputs.append(nbformat.v4.new_output(output_type='error', ename='ShellError', evalue=str(e), traceback=[]))

        self._write_notebook_object(nb) 
        return execution_successful
    
    def run_all_cells(self, stop_on_error: bool = True, restart: bool = False) -> int:
        """
        Runs all code cells in the notebook sequentially.
        Returns the index of the first failed cell, or None if all succeeded.
        """
        logger.debug("Running all code cells for session '%s'", self.name)
        if not self.notebook_file or not U.pexists(self.notebook_file):
            logger.warning("Notebook file not found for session '%s', cannot run cells", self.name)
            return None
            
        nb = self._read_notebook_object()
        if not nb: return None

        if restart:
            self.start_kernel()
            self.last_stop_index = 0

        failed_cell_idx = None
        for i, cell_data in enumerate(nb.cells):
            if i < self.last_stop_index: continue
            if cell_data.cell_type == 'code':
                logger.debug("Running cell %d in session '%s'", i, self.name)
                success = self.run_cell(i)
                if not success:
                    failed_cell_idx = i
                    logger.error("Error in cell %d (stop_on_error=%s)", i, stop_on_error)
                    if stop_on_error:
                        break
        overall_success = failed_cell_idx is None
        logger.debug("Finished running all cells for session '%s'. Overall success: %s", self.name, overall_success)
        if not overall_success:
            self.last_stop_index = 0 # rerun all when buggy to avoid buggy cells influence the others
        else:
            self.last_stop_index = len(nb.cells) # if no error, update it, no need to rerun the cells
        if restart:
            self.shutdown()
        return failed_cell_idx

    def shutdown(self):
        """Shuts down all resources for this session (server and kernel)."""
        logger.debug("Initiating full shutdown for session '%s'", self.name)
        self.shutdown_server()
        self.shutdown_kernel()
        logger.debug("Full shutdown for session '%s' completed", self.name)


class JupyterSandbox:
    project_root: str

    def __init__(self, config: Dict[str, Any], path: Optional[str] = None, verbose: bool = False):
        self.project_root = config['project_root']
        self.config = config
        self.sandbox_dir = path if path else U.pjoin(U.TMP_DIR, 'sandbox', config['name'])
        logger.debug("Initializing JupyterSandbox in: %s", self.sandbox_dir)
        self.session_dir = U.pjoin(self.sandbox_dir, 'sessions')
        self.active_sessions: Dict[str, JupyterSession] = {}

    def new_session(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> JupyterSession:
        metadata = (metadata or {}).copy()
        # project_root is the root directory of the project, it is used to load the proxy 
        proxy_cfg = metadata.get('proxy', {})
        proxy_cfg['activate_proxies'] = proxy_cfg.get('activate_proxies', self.config['activate_proxies'])
        proxy_cfg['cutoff_date'] = proxy_cfg.get('cutoff_date', None)
        proxy_cfg['deploy_mode'] = proxy_cfg.get('deploy_mode', False)
        metadata['project_root'] = self.project_root
        metadata['proxy'] = proxy_cfg
        metadata.setdefault('autorun', self.config.get('autorun_sessions', False))

        session_name_base = name if name else dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+uuid.uuid4().hex[:6]
        session_name = session_name_base
        session_path = path if path else U.pjoin(self.session_dir, session_name)

        logger.debug("Creating new session '%s' in directory: %s", session_name, session_path)
        U.mkdirs(session_path)
        notebook_file = U.pjoin(session_path, f"{session_name}.ipynb")
        if U.pexists(notebook_file):
            raise Exception(f"Session '{session_name}' already exists in {session_path}")
        sess = JupyterSession(name=session_name, dir=session_path, metadata=metadata, notebook_file=notebook_file)
        sess.init_session()
        sess._ensure_notebook_file(create=True) # Create the .ipynb file immediately
        
        # Save session metadata
        U.save_json(U.pjoin(session_path, f'{session_name}_meta.json'), sess.to_dict())
        self.active_sessions[session_name] = sess # Track it as active
        return sess
    

    def get_session(self, session_name: str, create: bool = True, metadata: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Optional[JupyterSession]: # Changed default create to False
        if session_name in self.active_sessions:
            return self.active_sessions[session_name]
        
        session_path = path if path else U.pjoin(self.session_dir, session_name)
        meta_file = U.pjoin(session_path, f'{session_name}_meta.json')
        
        if U.pexists(meta_file):
            logger.debug("Loading session '%s' from %s", session_name, meta_file)
            sess_data = U.load_json(meta_file)
            sess = JupyterSession.from_dict(sess_data)
            self.active_sessions[session_name] = sess # Add to active if loaded
            return sess
        elif create:
            logger.debug("Session '%s' not found, creating new", session_name)
            return self.new_session(name=session_name, metadata=metadata, path=session_path)
        else:
            logger.debug("Session '%s' not found and create=False", session_name)
            return None


    def shutdown_session_resources(self, session_name: str):
        if session_name in self.active_sessions:
            session = self.active_sessions[session_name]
            session.shutdown()
            logger.debug("Resources for session '%s' shut down", session_name)
        else:
            logger.warning("Session '%s' not found in active sessions for resource shutdown", session_name)

    def delete_session_completely(self, session_name: str):
        """Shuts down resources and removes session from disk."""
        logger.debug("Attempting to completely delete session '%s'", session_name)
        if session_name in self.active_sessions:
            session = self.active_sessions[session_name]
            session.shutdown() # Ensure server/kernel are off
            session_path = session.dir
            del self.active_sessions[session_name]
        else:
            # If not active, construct path from name
            session_path = U.pjoin(self.session_dir, session_name)

        if U.pexists(session_path):
            try:
                U.rmtree(session_path)
                logger.debug("Deleted session directory: %s", session_path)
            except Exception as e:
                logger.error("Error deleting session directory %s: %s", session_path, e)
        else:
            logger.warning("Session directory %s not found for deletion", session_path)


    def shutdown_all_sessions_resources(self):
        logger.debug("Shutting down resources for all active Jupyter sessions")
        for name in list(self.active_sessions.keys()):
            self.shutdown_session_resources(name)
        logger.debug("Resources for all active sessions shut down")
