"""Unified CLI: python main.py train|predict|evaluate [options]."""
import runpy,sys
from pathlib import Path
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in ('train','predict','evaluate'):
        raise SystemExit('Usage: python main.py {train|predict|evaluate} [arguments]')
    cmd=sys.argv.pop(1)
    sys.argv[0]=str(Path(__file__).parent/f'{cmd}.py')
    runpy.run_path(sys.argv[0],run_name='__main__')
