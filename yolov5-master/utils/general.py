# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
import shapely
import shapely.geometry
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

# Import 'ultralytics' package or install if if missing
try:
    import ultralytics

    assert hasattr(ultralytics, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    os.system('pip install -U ultralytics')
    import ultralytics

# 检查c语言插件加载
try:
    # if error in importing polygon_inter_union_cuda, polygon_b_inter_union_cuda, please cd to ./iou_cuda and run "python setup.py install"
    from polygon_inter_union_cuda import polygon_inter_union_cuda, polygon_b_inter_union_cuda
    polygon_inter_union_cuda_enable = True
    polygon_b_inter_union_cuda_enable = True
except Exception as e:
    print(f'Warning: "polygon_inter_union_cuda" and "polygon_b_inter_union_cuda" are not installed.')
    print(f'The Exception is: {e}.')
    polygon_inter_union_cuda_enable = False
    polygon_b_inter_union_cuda_enable = False

from ultralytics.utils.checks import check_requirements

from utils import TryExcept, emojis
from utils.downloads import curl_download, gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
DATASETS_DIR = Path(os.getenv('YOLOv5_DATASETS_DIR', ROOT.parent / 'datasets'))  # global datasets directory
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress verbose TF compiler warnings in Colab


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return bool(re.search('[\u4e00-\u9fff]', str(s)))


def is_colab():
    # Is environment a Google Colab instance?
    return 'google.colab' in sys.modules


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False


def is_kaggle():
    # Is environment a Kaggle Notebook?
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_docker() -> bool:
    """Check if the process runs inside a docker container."""
    if Path('/.dockerenv').exists():
        return True
    try:  # check if docker is in control groups
        with open('/proc/self/cgroup') as file:
            return any('docker' in line for line in file)
    except OSError:
        return False


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


LOGGING_NAME = 'yolov5'


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level, }},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False, }}})


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()  # Ultralytics settings dir


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class Timeout(contextlib.ContextDecorator):
    # YOLOv5 Timeout class. Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith('__')]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def file_age(path=__file__):
    # Return days since last file update
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket

    def run_once():
        # Check once
        try:
            socket.create_connection(('1.1.1.1', 443), 5)  # check host accessibility
            return True
        except OSError:
            return False

    return run_once() or run_once()  # check twice to increase robustness to intermittent connectivity issues


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo='ultralytics/yolov5', branch='master'):
    # YOLOv5 status check, recommend 'git pull' if code is out of date
    url = f'https://github.com/{repo}'
    msg = f', for updates see {url}'
    s = colorstr('github: ')  # string
    assert Path('.git').exists(), s + 'skipping check (not a git repository)' + msg
    assert check_online(), s + 'skipping check (offline)' + msg

    splits = re.split(pattern=r'\s', string=check_output('git remote -v', shell=True).decode())
    matches = [repo in s for s in splits]
    if any(matches):
        remote = splits[matches.index(True) - 1]
    else:
        remote = 'ultralytics'
        check_output(f'git remote add {remote} {url}', shell=True)
    check_output(f'git fetch {remote}', shell=True, timeout=5)  # git fetch
    local_branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
    n = int(check_output(f'git rev-list {local_branch}..{remote}/{branch} --count', shell=True))  # commits behind
    if n > 0:
        pull = 'git pull' if remote == 'origin' else f'git pull {remote} {branch}'
        s += f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use '{pull}' or 'git clone {url}' to update."
    else:
        s += f'up to date with {url} ✅'
    LOGGER.info(s)


@WorkingDirectory(ROOT)
def check_git_info(path='.'):
    # YOLOv5 git info check, return {remote, branch, commit}
    check_requirements('gitpython')
    import git
    try:
        repo = git.Repo(path)
        remote = repo.remotes.origin.url.replace('.git', '')  # i.e. 'https://github.com/ultralytics/yolov5'
        commit = repo.head.commit.hexsha  # i.e. '3134699c73af83aac2a481435550b968d5792c0d'
        try:
            branch = repo.active_branch.name  # i.e. 'main'
        except TypeError:  # not on any branch
            branch = None  # i.e. 'detached HEAD' state
        return {'remote': remote, 'branch': branch, 'commit': commit}
    except git.exc.InvalidGitRepositoryError:  # path is not a git dir
        return {'remote': None, 'branch': None, 'commit': None}


def check_python(minimum='3.8.0'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def check_img_size(imgsz, s=32, floor=0):
    # 验证图像大小是每个维度中步长的倍数
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_imshow(warn=False):
    # Check if environment supports image displays
    try:
        assert not is_jupyter()
        assert not is_docker()
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


def check_suffix(file='yolov5s.pt', suffix=('.pt', ), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    elif file.startswith('clearml://'):  # ClearML Dataset ID
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:  # search
        files = []
        for d in 'data', 'models', 'utils':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_font(font=FONT, progress=False):
    # Download font to CONFIG_DIR if necessary
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_dataset(data, autodownload=True):
    # Download, check and/or unzip dataset if not found locally

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f'{DATASETS_DIR}/{Path(data).stem}', unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in 'train', 'val', 'names':
        assert k in data, emojis(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data['names'], (list, tuple)):  # old array format
        data['names'] = dict(enumerate(data['names']))  # convert to dict
    assert all(isinstance(k, int) for k in data['names'].keys()), 'data.yaml names keys must be integers, i.e. 2: car'
    data['nc'] = len(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or '')  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info('\nDataset not found ⚠️, missing paths %s' % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception('Dataset not found ❌')
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f'Downloading {s} to {f}...')
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # create root
                unzip_file(f, path=DATASETS_DIR)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = subprocess.run(s, shell=True)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf', progress=True)  # download fonts
    return data  # dictionary


def check_amp(model):
    # Check PyTorch Automatic Mixed Precision (AMP) functionality. Return True on correct operation
    from models.common import AutoShape, DetectMultiBackend

    def amp_allclose(model, im):
        # All close FP32 vs AMP results
        m = AutoShape(model, verbose=False)  # model
        a = m(im).xywhn[0]  # FP32 inference
        m.amp = True
        b = m(im).xywhn[0]  # AMP inference
        return a.shape == b.shape and torch.allclose(a, b, atol=0.1)  # close to 10% absolute tolerance

    prefix = colorstr('AMP: ')
    device = next(model.parameters()).device  # get model device
    if device.type in ('cpu', 'mps'):
        return False  # AMP only used on CUDA devices
    f = ROOT / 'data' / 'images' / 'bus.jpg'  # image to check
    im = f if f.exists() else 'https://ultralytics.com/images/bus.jpg' if check_online() else np.ones((640, 640, 3))
    try:
        assert amp_allclose(deepcopy(model), im) or amp_allclose(DetectMultiBackend('yolov5n.pt', device), im)
        LOGGER.info(f'{prefix}checks passed ✅')
        return True
    except Exception:
        help_url = 'https://github.com/ultralytics/yolov5/issues/7908'
        LOGGER.warning(f'{prefix}checks failed ❌, disabling Automatic Mixed Precision. See {help_url}')
        return False


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    # Unzip a *.zip file to path/, excluding files containing strings in exclude list
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f'Downloading {url} to {f}...')
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
                else:
                    LOGGER.warning(f'❌ Failed to download {url}...')

        if unzip and success and (f.suffix == '.gz' or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f'Unzipping {f}...')
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(['tar', 'xf', f, '--directory', f.parent], check=True)  # unzip
            elif f.suffix == '.gz':
                subprocess.run(['tar', 'xfz', f, '--directory', f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern='[|@#!¡·$€%&()=?¿^*;:,¨´><+]', repl='_', string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'best_fitness', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
    evolve_csv = save_dir / 'evolve.csv'
    evolve_yaml = save_dir / 'hyp_evolve.yaml'
    keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
            subprocess.run(['gsutil', 'cp', f'{url}', f'{save_dir}'])  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv, skipinitialspace=True)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
                f'# Last generation: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) +
                '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

    # Print to screen
    LOGGER.info(prefix + f'{generations} generations finished, current result:\n' + prefix +
                ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
                                                                                         for x in vals) + '\n\n')

    if bucket:
        subprocess.run(['gsutil', 'cp', f'{evolve_csv}', f'{evolve_yaml}', f'gs://{bucket}'])  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


# rotate does not support cutout
def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)
    return crop


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


def imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename, img):
    try:
        cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(path, im):
    imshow_(path.encode('unicode_escape').decode(), im)


if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

# Variables ------------------------------------------------------------------------------------------------------------


# Ancillary functions with polygon anchor boxes-------------------------------------------------------------------------------------------

def xyxyxyxyn2xyxyxyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized xyxyxyxy or segments into pixel xyxyxyxy or segments
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = w * x[:, 0::2] + padw  # all x
    y[:, 1::2] = h * x[:, 1::2] + padh  # all y
    return y


def polygon_segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 polygon box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxyxyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside].reshape(-1, 1), y[inside].reshape(-1, 1)
    multipoint = shapely.geometry.MultiPoint(np.concatenate((x, y), axis=1))
    # polygon_box: x1, y1, x2, y2, x3, y3, x4, y4 (unnormalized)
    polygon_box = np.array(multipoint.minimum_rotated_rectangle.exterior.coords[:-1]).ravel()
    # polygon_box[0::2] = polygon_box[0::2].clip(0., width)
    # polygon_box[1::2] = polygon_box[1::2].clip(0., height)
    return polygon_box if any(x) else np.zeros((1, 8))  # xyxyxyxy


def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """

    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4, 2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """

    boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    if torch.cuda.is_available() and polygon_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0]  # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0]  # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0]  # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0]  # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                    w1, h1 = b1_x2[i] - b1_x1[i], b1_y2[i] - b1_y1[i] + eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU


def polygon_b_inter_union_cpu(boxes1, boxes2):
    """
        iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py;
        Boxes and Anchors having the same shape: nx8;
        Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
    """

    n = boxes1.shape[0]
    inter = torch.zeros(n, )
    union = torch.zeros(n, )
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
        polygon2 = shapely.geometry.Polygon(boxes2[i, :].view(4, 2)).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter[i] = polygon1.intersection(polygon2).area
                union[i] = polygon1.union(polygon2).area
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_bbox_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """

    boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    if torch.cuda.is_available() and polygon_b_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside inter_union_cuda must be torch.cuda.float, not double type or half type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_b_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_b_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0
    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes10 and boxes20
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0]  # n,
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0]  # n,
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0]  # n,
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0]  # n,
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                iou -= rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                iou -= (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            iou -= (c_area - union) / c_area  # GIoU
    return iou  # IoU


def polygon_nms_kernel(x, iou_thres):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9]
        Return the selected indices
    """

    unique_labels = x[:, 9].unique()
    _, scores_sort_index = torch.sort(x[:, 8], descending=True)
    x = x[scores_sort_index]
    indices = scores_sort_index
    selected_indices = []

    # Iterate through all predicted classes
    for unique_label in unique_labels:
        x_ = x[x[:, 9] == unique_label]
        indices_ = indices[x[:, 9] == unique_label]

        while x_.shape[0]:
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(x_) == 1: break
            # Compute the IOUs for all other the polygon boxes
            iou = polygon_box_iou(x_[0:1, :8], x_[1:, :8], device=x.device).view(-1)
            # Remove overlapping detections with IoU >= NMS threshold
            x_ = x_[1:][iou < iou_thres]
            indices_ = indices_[1:][iou < iou_thres]

    return torch.LongTensor(selected_indices)


def order_corners(boxes):
    """
        Sorted corners have the following restrictions:
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """

    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y)  # sort y
    x_sorted = torch.zeros_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        x_sorted[i] = x[i, y_indices[i]]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(y.shape[0]):
        y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()


# Ancillary functions with rotate anchor boxes-------------------------------------------------------------------------------------------

def xywhrm2xyxyxyxy(xywhrm):
    """
        xywhrm : shape (N, 6)
        Transform x,y,w,h,re,im to x1,y1,x2,y2,x3,y3,x4,y4
        Suitable for both pixel-level and normalized
    """
    is_array = isinstance(xywhrm, np.ndarray)
    if is_array:
        xywhrm = torch.from_numpy(xywhrm)

    x0, x1, y0, y1 = -xywhrm[:, 2:3] / 2, xywhrm[:, 2:3] / 2, -xywhrm[:, 3:4] / 2, xywhrm[:, 3:4] / 2
    xyxyxyxy = torch.cat((x0, y0, x1, y0, x1, y1, x0, y1), dim=-1).view(-1, 4, 2).contiguous()
    R = torch.zeros((xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype, device=xyxyxyxy.device)
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]

    xyxyxyxy = torch.matmul(xyxyxyxy, R).view(-1, 8).contiguous() + xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    return xyxyxyxy.cpu().numpy() if is_array else xyxyxyxy


def xyxyxyxy2xywhrm(xyxyxyxy):
    """
        xyxyxyxy : shape (N, 8)
        Transform x1,y1,x2,y2,x3,y3,x4,y4 to cx,cy,w,h,re,im
        Suitable for both pixel-level and normalized
    """
    is_array = isinstance(xyxyxyxy, np.ndarray)
    xyxyxyxy = order_corners(torch.from_numpy(xyxyxyxy)) if is_array else order_corners(xyxyxyxy.clone())

    w = torch.max(((xyxyxyxy[:, 2] - xyxyxyxy[:, 0]) ** 2 + (xyxyxyxy[:, 3] - xyxyxyxy[:, 1]) ** 2).sqrt(),
                  ((xyxyxyxy[:, 6] - xyxyxyxy[:, 4]) ** 2 + (xyxyxyxy[:, 7] - xyxyxyxy[:, 5]) ** 2).sqrt())
    h = torch.max(((xyxyxyxy[:, 4] - xyxyxyxy[:, 2]) ** 2 + (xyxyxyxy[:, 5] - xyxyxyxy[:, 3]) ** 2).sqrt(),
                  ((xyxyxyxy[:, 6] - xyxyxyxy[:, 0]) ** 2 + (xyxyxyxy[:, 7] - xyxyxyxy[:, 1]) ** 2).sqrt())

    cx, cy = xyxyxyxy[:, 0::2].mean(dim=-1), xyxyxyxy[:, 1::2].mean(dim=-1)
    theta_tan = (xyxyxyxy[:, 3] - xyxyxyxy[:, 1]) / (xyxyxyxy[:, 2] - xyxyxyxy[:, 0])
    re = theta_tan.atan().cos()
    im = theta_tan.atan().sin()

    xywhrm = torch.stack((cx, cy, w, h, re, im), dim=-1)
    return xywhrm.cpu().numpy() if is_array else xywhrm


def xywhn2xywh(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x, y, w, h] pixel-level
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # center x
    y[:, 1] = h * x[:, 1] + padh  # center y
    y[:, 2] = w * x[:, 2]  # width
    y[:, 3] = h * x[:, 3]  # height
    return y


def rotate_segments2boxes(segments, img_shapes=None):
    # Convert segment labels to rotate box labels, i.e. (xy1, xy2, ...) to rotated boxes (x, y, w, h, re, im)
    boxes = []
    img_shapes = [None] * len(segments) if img_shapes is None else img_shapes
    for segment, img_shape in zip(segments, img_shapes):
        polygon_box = polygon_segment2box(segment) if img_shape is None else polygon_segment2box(segment, img_shape[1],
                                                                                                 img_shape[0])
        boxes.append(xyxyxyxy2xywhrm(polygon_box)[0])  # list with item of xywhrm
    return np.array(boxes)  # numpy array with row of xywhrm


# Rescale coords (x, y, w, h, re, im) from img1_shape to img0_shape
def rotate_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, img0_shape[1])  # cx
    coords[:, 1].clamp_(0, img0_shape[0])  # cy
    coords[:, 2].clamp_(0, img0_shape[1])  # width
    coords[:, 3].clamp_(0, img0_shape[0])  # height
    return coords


def rotate_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of rotated boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx6, boxes2 is mx6
    """

    boxes1_xyxyxyxy = xywhrm2xyxyxyxy(boxes1)
    boxes2_xyxyxyxy = xywhrm2xyxyxyxy(boxes2)
    return polygon_box_iou(boxes1_xyxyxyxy, boxes2_xyxyxyxy, GIoU, DIoU, CIoU, eps, device)  # IoU


def rotate_bbox_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of rotated boxes for class Rotate_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """

    boxes1_xyxyxyxy = xywhrm2xyxyxyxy(boxes1)
    boxes2_xyxyxyxy = xywhrm2xyxyxyxy(boxes2)
    return polygon_bbox_iou(boxes1_xyxyxyxy, boxes2_xyxyxyxy, GIoU, DIoU, CIoU, eps, device)  # IoU


# 旋转框非极大抑制实现
def rotate_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                               multi_label=False,
                               labels=(), max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results for rotated boxes
        Returns:  list of detections, on (n,8) tensor per image [x, y, w, h, re, im, conf, cls]
        数据统一进行归一化：夹角θ∈[-90,90] cosθ用sigmoid归一化、sinθ用tanh归一化
        x: 中心点横坐标
        y: 中心点纵坐标
        w: 矩形框宽
        h: 矩形框高
        re: 旋转夹角cosθ
        im: 旋转夹角sinθ
    """

    # 预测结果输出（batch_size, all_boxes, 6+1+num_class）
    assert not agnostic, "rotated boxes does not support agnostic"
    # 获得类别数
    nc = prediction.shape[2] - 7  # number of classes
    # 筛选满足置信度条件的框，第6列是所有框的置信度值
    xc = prediction[..., 6] > conf_thres  # confidence candidates

    # 检查不满足条件区间的置信度值和IOU值
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 3, 4096  # (pixels) minimum and maximum box width and height   预测物体宽度和高度的大小范围
    max_nms = 30000  # maximum number of boxes into polygon_nms_kernel, can increase this value 每个图像最多检测物体的个数
    time_limit = 10.0  # seconds to quit after  nms执行时间阈值 超过这个时间就退出了
    redundant = True  # require redundant detections    是否需要冗余的detections
    # 每个box可以有多个label(多标签)
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS 多个bounding box给它们一个权重进行融合  默认False

    t = time.time()  # 当前时间
    # 新建一个list 包含batchsize个空的output tensor
    output = [torch.zeros((0, 8), device=prediction.device)] * prediction.shape[0]
    # 一张图片一张图片的处理
    for xi, x in enumerate(prediction):  # image index, image inference 遍历所有的框
        # Apply constraints
        # 第一层过滤 虑除超小anchor标和超大anchor
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 6] = 0  # width-height

        # 第二层过滤 根据conf_thres虑除背景目标(obj_conf<conf_thres 0.1的目标 置信度极低的目标)
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 7), device=x.device)
            v[:, :6] = l[:, 1:7]  # box
            v[:, 6] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 7] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # Compute conf 计算conf_score
        # 利用numpy切片，首先将所有行全部选取出来，再将第7列即以后的元素取出，与第6列的元素相乘
        x[:, 7:] *= x[:, 6:7]  # conf_score = obj_conf * cls_conf

        # 修改一：原来YOLOv5对box的处理是将 xywh=>xyxy  即中心宽高=>左上右下，我们这里直接将预选框复制到box，即保持6参形式
        # 原来Box：(x1, y1, x2, y2)
        # 现在Box：(x, y, w, h, re, im)
        box = x[:, :6].clone()

        # Detections matrix nx8 (xywhrm, conf, cls)
        # Transfer sigmoid probabilities of classes (e.g. three classes [0.567, 0.907, 0.01]) to selected classes (1.0)
        if multi_label:
            # 第三轮过滤:针对每个类别score(obj_conf * cls_conf) > conf_thres
            i, j = (x[:, 7:] > conf_thres).nonzero(as_tuple=False).T
            # concat satisfied boxes (multi-label-enabled) along 0 dimension
            x = torch.cat((box[i], x[i, j + 7, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 7:].max(1, keepdim=True)  # 一个类别直接取分数最大类的即可
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class   是否只保留特定的类别  默认None  不执行这里
        if classes is not None:
            x = x[(x[:, 7:8] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes   如果经过第三轮过滤该feature map没有目标框了，就结束这轮直接进行下一张图
            continue
        elif n > max_nms:  # excess boxes   如果经过第三轮过滤该feature map还有很多框(>max_nms)   就需要排序
            x = x[x[:, 6].argsort(descending=True)[:max_nms]]  # sort by confidence

        # 修改二：
        # Rotate NMS 不支持 Batch NMS and Agnostic
        # x是x[:, :6]排序后的预测 , confidence x[:, 6], class x[:, 7]
        # X_是x_[:, :8]排序后的预测 , confidence x_[:, 8], class x_[:, 9]
        # 不能使用 torchvision.ops.nms, 该方法只处理水平对齐的方框
        x_ = torch.zeros((x.shape[0], 10), dtype=x.dtype, device=x.device)
        x_[:, 8:10] = x[:, 6:8]
        x_[:, :8] = xywhrm2xyxyxyxy(x[:, :6])
        i = polygon_nms_kernel(x_, iou_thres)  # polygon-NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            boxes = x[:, :6]
            # update boxes as boxes(i,6) = weights(i,n) * rotated boxes(n,6)
            iou = rotate_box_iou(boxes[i], boxes, device=prediction.device) > iou_thres  # iou matrix
            weights = iou * x[:, 6][None]  # rotated box weights
            x[i, :6] = torch.mm(weights, x[:, :6]).float() / weights.sum(1, keepdim=True)  # merged rotated boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        # 看下时间超没超时  超时没做完的就不做了
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output
