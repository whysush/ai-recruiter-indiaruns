# Submission checklist

Everything technical is done and validated. Only the items marked **YOU** below
need your real values (they can't be auto-filled).

## 1. Generate the ranked CSV (already valid)
```bash
python scripts/run_submission.py --candidates data/candidates.jsonl --out submission.csv
```
Produces 100 rows, self-runs the official validator → "Submission is valid."

## 2. YOU — name the file after your registered participant ID  (spec §2)
The upload file must be `<your_participant_id>.csv`. One command:
```bash
bash scripts/finalize_submission.sh team_xxxxx     # <- replace team_xxxxx with YOUR id
```
This writes `team_xxxxx.csv` and re-validates it. Upload that file.

## 3. YOU — fill the placeholders in `submission_metadata.yaml`  (spec §10.2)
Replace every `TODO`:
- `team_name`
- `primary_contact.name`, `.phone`   (email is already set)
- `team_members[].name` / `.email`   (one block per member)
- `github_repo`  → your real `https://github.com/<user>/<repo>`
- `sandbox_link` → either your hosted demo URL, or keep the GitHub `#sandbox` anchor
  (the Docker recipe in the README satisfies §10.5)

## 4. YOU — push the repo to GitHub
```bash
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main      # or 'master'
```
The repo already has clean, incremental history (8 commits) and includes the
committed embedding artifacts so the no-network Stage-3 reproduction works.

## 5. Portal upload (spec §10.2 fields to have ready)
Team name · contact name/email/phone · GitHub URL · sandbox link ·
AI tools = Claude · compute summary ("Local Linux, 16GB, Python 3.10, CPU-only") ·
team member list · methodology summary (already in `submission_metadata.yaml`).

---
### Quick reference — the three submission parts (spec §10)
| Part | Status |
|---|---|
| 10.1 The CSV (top-100) | ✅ generated + valid; **rename to your id** |
| 10.2 Portal metadata | ⬜ **YOU** fill `submission_metadata.yaml` + portal form |
| 10.3 Code repo (README, source, artifacts, deps, metadata) | ✅ complete; **push to GitHub** |
