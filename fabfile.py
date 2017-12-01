from fabric.api import *
from fabric.contrib.project import rsync_project

# the user to use for the remote commands
env.user = 'eliaslinux'
# the servers where the commands are executed
env.hosts = ['192.168.0.6']


def deploy():
    # figure out the package name and version
    with cd('/home/eliaslinux/bin/Retrieval'):
        rsync_project(local_dir='.', remote_dir='/home/eliaslinux/bin/Retrieval', exclude=['path_pre_process', 'corpus', '__pycache__'])


def exec(filename):
    with cd('/home/eliaslinux/bin/Retrieval'):
        run('/home/eliaslinux/InkaLabs/HVB/virtualenvs/retrieval-py3/bin/python %s.py' % filename)