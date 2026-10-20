THREE JOBS FOR AN AI TOOL

What this is
------------
All documents here are invented (invoices, receipts, credit notes); there is no real company, person or data. An AI tool with access to this folder does three jobs. Each job is done in its OWN new chat, because the tool works better with a clean start:
  A  writes 60 new evaluation documents, following a guide      (PROMPT_A.txt)
  B  labels 160 documents independently, following the same guide (PROMPT_B.txt)
  C  rewrites 235 generated documents in a more natural way        (PROMPT_C.txt)

What you need
-------------
- An AI tool that can read and write files in a folder and run commands (a coding agent).
- Python 3.9 or newer. Nothing needs to be installed.
- About two hours of the tool running in total. It works on its own.

How to do it
------------
1. Unzip the file and open the AI tool in THIS folder (the one with this README).
2. Start a NEW chat. Paste the whole text of PROMPT_A.txt. Wait until the tool says job A is done.
3. Start a NEW chat. Paste the whole text of PROMPT_B.txt. Wait until it says job B is done.
4. Start a NEW chat. Paste the whole text of PROMPT_C.txt. It ends with  python finish.py  and says when results.zip is ready.
5. Send me results.zip.
Do job A first. The order of B and C does not matter, but C must be last because it makes the zip.
If the tool stops (limit, closed session) during a job, start a new chat and paste the same prompt again. It continues where the files stop.

Folders and files
-----------------
README.txt                this file
PROMPT_A.txt, _B, _C      the three messages for the tool
GUIDE.md                  the booking rules (used by jobs A and B)
A_write/                  job A: WRITE_GUIDE.md, config.json, new_records.jsonl
B_label/                  job B: docs.jsonl, config.json, answers.jsonl
C_rewrite/                job C: RULES.md, tasks.jsonl, config.json, out/
tools/                    the check programs (the tool runs them)
finish.py                 the last step: checks everything and makes results.zip

Please
------
Use only the files in this folder. Do not add examples of your own and do not share the files.
