from subprocess import Popen, PIPE, CalledProcessError

with Popen('python main.py --control \'train\'', shell=True,stdout=PIPE, bufsize=1, universal_newlines=True) as p:
    for line in p.stdout:
        #传递给前端网页
        print(line, end='')

if p.returncode != 0:
    raise CalledProcessError(p.returncode, p.args)