import sys
import io
import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import mlflow
from mlflow.tracking import MlflowClient

os.environ['MLFLOW_SUPPRESS_EMOJIS'] = '1'
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)
# ----------------------------------------------------------------------------------------------------------------------
class MLFlower(object):
    def __init__(self, host, port, username_mlflow=None, password_mlflow=None,remote_storage_folder=None, username_ssh=None, ppk_key_path=None,password_ssh=None, local_storage_folder=None):

        if not self.check_is_available(host, port, username_mlflow, password_mlflow):
            self.is_available = False
            return

        if remote_storage_folder is not None:
            self.host_ssh = self.construct_host_ssh(host)

        if local_storage_folder is None:
            local_storage_folder = os.path.abspath('./mlflow_artifacts')
        else:
            local_storage_folder = os.path.abspath(local_storage_folder)

        os.makedirs(local_storage_folder, exist_ok=True)
        artifact_root = f"file:///{local_storage_folder.replace(chr(92), '/')}"
        os.environ['MLFLOW_ARTIFACT_ROOT'] = artifact_root
        os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = artifact_root
        mlflow.set_tracking_uri(f'{host}:{port}')
        self.is_available = True

        if username_mlflow is not None and password_mlflow is not None:
            os.environ['MLFLOW_TRACKING_USERNAME'] = username_mlflow
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password_mlflow

        self.host = host  # ✓ Add this
        self.remote_host = host
        self.remote_port = port
        self.local_storage_folder = local_storage_folder  # ✓ Add this

        self.username_ssh = username_ssh
        self.ppk_key_path = ppk_key_path
        self.password_ssh = password_ssh
        self.remote_storage_folder = remote_storage_folder

        # print(f"Artifact storage configured: {local_storage_folder}")
        # print(f'ML Flow server available at {host}:{port} OK')
        # print(f'Tracking URI: {host}:{port}')
        # print(f'Local Storage: {local_storage_folder}')
        # if remote_storage_folder:
        #     print(f'  Remote Storage: {remote_storage_folder}')
        return

    # ---------------------------------------------------------------------------------------------------------------------
    def construct_host_ssh(self,host):
        host_ssh = host
        if host_ssh.startswith('http://'):host_ssh = host_ssh[7:]
        elif host_ssh.startswith('https://'):host_ssh = host_ssh[8:]
        return host_ssh
    # ---------------------------------------------------------------------------------------------------------------------
    def check_is_available(self,host,port,username, password):

        if host is None or port is None:
            return False

        try:
            auth = HTTPBasicAuth(username, password) if username is not None and password is not None else None
            response = requests.get(f'{host}:{port}', auth=auth, timeout=5)
            if response.status_code == 200:
                result = True
            else:
                result = False

        except requests.exceptions.RequestException as e:
            result = False

        if result is False:
            print(f'ML FLow server unavailable at {host}:{port} with user {username}')

        return result
    # ---------------------------------------------------------------------------------------------------------------------
    def check_ssh(self,host_ssh,username,password):

        if password is None:
            return False


        command = 'sshpass -p %s ssh -o StrictHostKeyChecking=no -p %d %s@%s exit'%(password,int(22),username,host_ssh)
        response = os.system(command)
        result = (response == 0)

        if result is False:
            print(f'SSH unavailable: {username}@{host_ssh}')

        return result
    # ---------------------------------------------------------------------------------------------------------------------
    def scp_file_to_remote(self,local_file_path, remote_folder):
        filename_only = local_file_path.split('/')[-1]

        if self.ppk_key_path is not None:
            command = f'scp -o StrictHostKeyChecking=no -P {int(22)} -i {self.ppk_key_path} {local_file_path} {self.username_ssh}@{self.host_ssh}:{remote_folder}{filename_only}'
        elif self.password_ssh is not None:
            command = f'sshpass -p {self.password_ssh} ssh -o StrictHostKeyChecking=no -p {int(22)} {self.username_ssh}@{self.host_ssh} mkdir -p {remote_folder}'
            os.system(command)
            command  = f'sshpass -p {self.password_ssh} scp -o StrictHostKeyChecking=no {local_file_path} {self.username_ssh}@{self.host_ssh}:{remote_folder}{filename_only}'
            os.system(command)

        return
    # ---------------------------------------------------------------------------------------------------------------------
    def get_experiment_id(self,experiment_name, create=True):
        experiement = mlflow.get_experiment_by_name(experiment_name)
        if experiement is not None:
            experiment_id = experiement.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name) if create else None

        return experiment_id
# ---------------------------------------------------------------------------------------------------------------------
    def get_run_params(self,run_id):
        return mlflow.get_run(run_id).data.params
    # ---------------------------------------------------------------------------------------------------------------------
    def get_run_metrics(self,run_id):
        return mlflow.get_run(run_id).data.metrics
    # ---------------------------------------------------------------------------------------------------------------------
    def get_run_artifact_filenames(self,run_id):
        client = MlflowClient()
        artifact_uri = client.get_run(run_id).info.artifact_uri
        artifacts = client.list_artifacts(run_id)
        filenames = [artifact_uri + artifact.path for artifact in artifacts]
        return filenames
    # ---------------------------------------------------------------------------------------------------------------------
    def download_artifacts(self,local_dir,run_id):
        client = MlflowClient()
        client.download_artifacts(run_id, "", local_dir)
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def delete_run(self,run_id):
        mlflow.delete_run(run_id)
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def update_run(self,run_id,metrics={},artifacts=[]):
        mlflow.end_run()
        with mlflow.start_run(run_id=run_id) as run:
            for k, v in metrics.items():mlflow.log_metric(k, v)
            for local_file_path in artifacts:
                if self.username_ssh is not None:
                    remote_folder = self.remote_storage_folder + mlflow.get_artifact_uri().split(':/')[1] + '/'
                    self.scp_file_to_remote(local_file_path, remote_folder)
                else:
                    mlflow.log_artifact(local_file_path)
            mlflow.end_run()
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def set_tracking_local(self,folder_out):
        #Runs are recorded locally
        mlflow.set_tracking_uri(folder_out)
        mlflow.set_registry_uri(folder_out)
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def set_tracking_remote(self,connection_string):
        # Runs are recorded remotely, Database is encoded as <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
        mlflow.set_tracking_uri(connection_string)
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def get_uris(self):

        print('is_tracking_uri_set:', mlflow.tracking.is_tracking_uri_set())
        print('tracking.get_tracking_uri:', mlflow.tracking.get_tracking_uri())
        print('registry_uri:',mlflow.get_registry_uri())
        print('tracking_uri:',mlflow.get_tracking_uri())

        artifact_uri = mlflow.get_artifact_uri()
        print('artifact_uri:',artifact_uri)

        return artifact_uri
    # ---------------------------------------------------------------------------------------------------------------------
    def save_experiment(self, experiment_name, params={}, metrics={}, artifacts=[]):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            mlflow.end_run()
        finally:
            sys.stdout = old_stdout

        with mlflow.start_run(experiment_id=self.get_experiment_id(experiment_name, create=True),run_name=pd.Timestamp.now().strftime('%Y-%b-%d %H:%M:%S')) as run:
            params['run_id'] = run.info.run_id

            for k, v in params.items():mlflow.log_param(k, v)
            for k, v in metrics.items():mlflow.log_metric(k, v)
            for local_file_path in artifacts:
                if self.remote_storage_folder is not None:
                    remote_folder = self.remote_storage_folder + mlflow.get_artifact_uri().split(':/')[1] + '/'
                    self.scp_file_to_remote(local_file_path, remote_folder)
                else:
                    mlflow.log_artifact(local_file_path)

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                mlflow.end_run()
            finally:
                sys.stdout = old_stdout

        print(f"Run saved: {params['run_id']}")
        return params['run_id']
    # ---------------------------------------------------------------------------------------------------------------------