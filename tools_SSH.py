import os
import yaml
import paramiko
# --------------------------------------------------------------------------------------------------------------------
class SSH_Client(object):
    def __init__(self,filename_config):
        self.load_private_config(filename_config)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_private_config(self,filename_in):

        if filename_in is None:return None
        if not os.path.isfile(filename_in):return None

        with open(filename_in, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.remote_ssh_host = config['ssh']['host']
            self.remote_ssh_port = config['ssh']['port']
            self.remote_ssh_username = config['ssh']['user']
            self.ssh_ppk_key_path = config['ssh']['ppk_key_path']

        return
# ----------------------------------------------------------------------------------------------------------------------
    def scp_file_to_remote(self, local_file_path, remote_file_path):
        command = 'scp -o StrictHostKeyChecking=no -P %d -i %s %s %s@%s:%s' % (int(self.remote_ssh_port), self.ssh_ppk_key_path, local_file_path, self.remote_ssh_username, self.remote_ssh_host, remote_file_path)
        os.system(command)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def scp_file_from_remote(self, remote_file_path, local_file_path):
        command = 'scp -o StrictHostKeyChecking=no -P %d -i %s %s@%s:%s %s' % (int(self.remote_ssh_port), self.ssh_ppk_key_path, self.remote_ssh_username, self.remote_ssh_host, remote_file_path, local_file_path)
        os.system(command)
        return
# ---------------------------------------------------------------------------------------------------------------------
    def execure_command(self,command):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.remote_ssh_host,port=self.remote_ssh_port,username=self.remote_ssh_username,key_filename=self.ssh_ppk_key_path)
        stdin, stdout, stderr = client.exec_command(command)
        stdout_decoded = stdout.read().decode("utf-8")
        return
# ---------------------------------------------------------------------------------------------------------------------