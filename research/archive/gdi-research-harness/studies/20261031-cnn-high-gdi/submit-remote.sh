set -euo pipefail
for name in high90 rare; do
  root="/abr/RESEARCH_USER/gdi-cnn-${name}-20261031"
  cd "$root"
  test ! -e submission.txt
  tar -xzf code-only.tar.gz
  cp -- /abr/RESEARCH_USER/gdi-cnn-multiwindow-20261030/data.npz "$root/data.npz"
  /usr/bin/python3.8 -c 'import json,hashlib,pathlib; m=json.load(open("manifest.json")); assert all(hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest()==v for k,v in m.items()); print("manifest verified")'
  sbatch --parsable --job-name="gdi-cnn-${name}" --account=iucp --partition=p02 --qos=normal --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=18:00:00 --exclude=DIS03,DIS04,DIS05,DIS06 --output=logs/worker-%j.log --wrap='/usr/bin/python3.8 -u -B worker.py' | tee submission.txt
done
