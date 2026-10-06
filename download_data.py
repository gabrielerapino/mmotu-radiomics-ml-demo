"""Download the publicly linked MMOTU folder, subject to Google Drive availability."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import gdown
from PIL import Image

IMAGE_FOLDER_ID="1UGWeRjf2uQDdJR4V0dmsgMCm9xFE5kNU"
ANNOTATION_FOLDER_ID="1By888Q_DmDsVb5frzNS_eFbUgifvJhfl"
LABEL_IDS={"train_cls.txt":"1Em2TK2iXpmZ8ypWTAR2G5hH5cc1e4bMo",
           "val_cls.txt":"1BLt9XiVmqVMf4HJVEC4z8hJrMr6uUeC3"}
DEST=Path("data/raw/OTU_2d")

if __name__=="__main__":
    DEST.mkdir(parents=True,exist_ok=True)
    items=[]
    try:
        for folder_id, subfolder in [(IMAGE_FOLDER_ID,"images"),(ANNOTATION_FOLDER_ID,"annotations")]:
            folder_items=gdown.download_folder(id=folder_id,output=str(DEST/subfolder),quiet=True,
                                                use_cookies=False,skip_download=True,timeout=30)
            items.extend(folder_items)
    except Exception as exc:
        status={"success":False,"reason":f"Google Drive folder listing failed: {type(exc).__name__}: {exc}"}
        (DEST/"download_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        raise SystemExit(status["reason"])
    chosen=[]
    for item in items:
        rel=Path(item.local_path)
        name=rel.name.lower()
        is_image=rel.parent.name.lower()=="images" and rel.suffix.lower() in {".jpg",".jpeg",".png"}
        is_mask=rel.parent.name.lower()=="annotations" and name.endswith("_binary.png") and not name.endswith("_binary_binary.png")
        if is_image or is_mask:
            chosen.append(item)
    def fetch(item):
        target=Path(item.local_path)
        if target.is_file() and target.stat().st_size>0:
            try:
                with Image.open(target) as image: image.verify()
                return str(target)
            except Exception:
                pass
        target.parent.mkdir(parents=True,exist_ok=True)
        return gdown.download(id=item.id,output=str(target),quiet=True,use_cookies=False,retries=2)
    failures=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures={pool.submit(fetch,item):item for item in chosen}
        for future in as_completed(futures):
            item=futures[future]
            try:
                if not future.result(): failures.append(item.path)
            except Exception:
                failures.append(item.path)
    for filename,file_id in LABEL_IDS.items():
        target=DEST/filename
        try:
            if target.is_file() and target.stat().st_size>0:
                rows=[line.split() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
                if rows and all(len(row)==2 and row[1].isdigit() for row in rows):
                    continue
            result=gdown.download(id=file_id,output=str(target),quiet=True,use_cookies=False,retries=2)
            if not result: failures.append(filename)
        except Exception:
            failures.append(filename)
    if failures:
        reason=(f"Google Drive did not resolve {len(failures)} public file links during selective download. "
                "The folder listing was visible, but gdown reported that file permissions or Drive access limits may prevent download.")
        status={"success":False,"selected_file_count":len(chosen),"failed_files":failures[:20],"reason":reason}
        (DEST/"download_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        raise SystemExit(reason+" Retry later or download the files manually.")
    (DEST/"download_status.json").write_text(json.dumps({"success":True,"downloaded_file_count":len(chosen)+len(LABEL_IDS)},indent=2),encoding="utf-8")
    print(f"Downloaded {len(chosen)} images, masks and label-list files to {DEST}")
