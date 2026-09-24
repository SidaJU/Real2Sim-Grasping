"""T1.1  List available YCB assets and confirm their exact paths.

Run in Isaac Sim > Window > Script Editor.
Output: assets root URL + list of YCB object directory entries.
"""

from isaacsim.storage.native import get_assets_root_path
import omni.client

root = get_assets_root_path()
print("assets root:", root)

for sub in ["/Isaac/Props/YCB/Axis_Aligned/", "/Isaac/Props/YCB/"]:
    url = root + sub
    status, entries = omni.client.list(url)   # tuple unpacking: (result_code, entries)
    if status == omni.client.Result.OK:
        print("EXISTS:", sub)
        for e in entries[:25]:
            print("   ", e.relative_path)
    else:
        print("NOT FOUND:", sub)
