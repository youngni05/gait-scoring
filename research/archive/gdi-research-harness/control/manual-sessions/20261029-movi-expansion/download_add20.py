"""Reuse the checked downloader with the approved video-only batch manifest."""
from pathlib import Path
import download_subset as worker

worker.CONTROL=worker.ROOT/'control'/'add20'
worker.MANIFEST=worker.CONTROL/'download-manifest.json'
if __name__=='__main__':worker.main()
