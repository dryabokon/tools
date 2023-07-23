import os
import pandas as pd
import mlflow
# ---------------------------------------------------------------------------------------------------------------------
folder_out = './mlruns'
# ----------------------------------------------------------------------------------------------------------------------
class MLFlower(object):
    def __init__(self,host,port,remote_storage_folder=None,remote_username=None,ppk_key_path=None):
        mlflow.set_tracking_uri('http://%s:%s/' % (host, port))
        self.remote_host = host
        self.remote_port = port
        self.remote_username = remote_username
        self.ppk_key_path = ppk_key_path
        self.remote_storage_folder = remote_storage_folder
        return
    # ---------------------------------------------------------------------------------------------------------------------
    def scp_file_to_remote(self,local_file_path, remote_file_path, remote_host, remote_username, ppk_key_path, remote_port=22):
        command = 'scp -o StrictHostKeyChecking=no -P %d -i %s %s %s@%s:%s'%(int(remote_port),ppk_key_path,local_file_path,remote_username,remote_host,remote_file_path)
        #print(command)
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
    def save_experiment(self,experiment_name,params,metrics,artifacts):
        mlflow.end_run()

        ts = pd.Timestamp.now().strftime('%Y-%b-%d %H:%M:%S')

        with mlflow.start_run(experiment_id=self.get_experiment_id(experiment_name, create=True), run_name=ts) as run:
            print('exp_id:', run.info.experiment_id)
            print('run_id:',run.info.run_id)
            for k,v in  params.items():mlflow.log_param(k, v)
            for k, v in metrics.items():mlflow.log_metric(k, v)
            if self.remote_username is not None:
                for local_file_path in artifacts:
                    remote_file_path = mlflow.get_artifact_uri()
                    remote_file_path = self.remote_storage_folder + remote_file_path[1:] + '/'+local_file_path.split('/')[-1]
                    self.scp_file_to_remote(local_file_path, remote_file_path, self.remote_host, self.remote_username, self.ppk_key_path)
            mlflow.end_run()
        return
    # ---------------------------------------------------------------------------------------------------------------------