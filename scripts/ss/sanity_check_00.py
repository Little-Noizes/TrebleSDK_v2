from pathlib import Path
import sys, datetime
from treble_tsdk import treble

LOG = Path("logs/sanity_check.log"); LOG.parent.mkdir(parents=True, exist_ok=True)
def log(s): print(s); LOG.write_text((LOG.read_text() if LOG.exists() else "")+s+"\n", encoding="utf-8")

def must_exist(p: Path, label: str):
    if not p.exists(): log(f"[ERR] Missing {label}: {p}"); sys.exit(2)
    log(f"[OK ] Found {label}: {p.resolve()}")

def main():
    log(f"== sanity_check {datetime.datetime.now().isoformat(timespec='seconds')} ==")
    try:
        tsdk = treble.TSDK()  # standard init
        log("[OK ] Treble SDK initialised")
        me = tsdk.user_info.email if hasattr(tsdk, "user_info") else "<unknown>"
        log(f"[OK ] Logged in as: {me}")
        projects = tsdk.list_my_projects()
        log(f"[OK ] My projects count: {len(projects)}")
    except Exception as e:
        log(f"[ERR] Treble init/auth failed: {e}"); sys.exit(3)

    must_exist(Path("configs/classroom1.yaml"), "YAML")
    must_exist(Path("input/directivity/GenelecOy-8030.cf2"), "CF2")
    log("[OK ] Sanity check passed")

if __name__ == "__main__": main()
