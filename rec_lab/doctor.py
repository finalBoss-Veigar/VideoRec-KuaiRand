"""Read-only environment diagnosis; no installation or config mutation."""
import importlib
import platform
import sys
from pathlib import Path

def doctor():
    result={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'cwd':str(Path.cwd()),'project_root':str(Path(__file__).resolve().parent.parent),'packages':{}}
    for name in ['numpy','pandas','sklearn','yaml','matplotlib','torch']:
        try:
            module=importlib.import_module(name)
            result['packages'][name]={'installed':True,'version':getattr(module,'__version__','unknown')}
            if name=='torch':result['cuda_available']=module.cuda.is_available()
        except ImportError:
            result['packages'][name]={'installed':False}
    result['ready']=all(v['installed'] for v in result['packages'].values())
    return result
