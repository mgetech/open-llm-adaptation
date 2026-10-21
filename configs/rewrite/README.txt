REWRITE TASK

What this is
------------
A program generated a set of invoices, receipts and credit notes. All of them are invented; there is no real company, person or data in them. Their text is too regular. An AI model with access to this folder rewrites the text of each document in a more natural way and keeps every fact the same. A small check program rejects any rewrite where a fact changed, and asks for it again.

What you need
-------------
- An AI tool that can read and write files in a folder and run commands (a coding agent). It must be able to run Python 3.9 or newer. Nothing needs to be installed.
- About a few hours of the tool running. It works on its own after the pilot.

Files in this folder
--------------------
README.txt           this file
RULES.md             the rules for the AI model (it reads them itself)
PROMPT_1_PILOT.txt   the first message: a test with 40 documents
PROMPT_2_FULL.txt    the second message: all the other documents
rewrite.py           the check program (the AI model runs it)
config.json          settings of the check program
tasks.jsonl          all the documents to rewrite
pilot.jsonl          the documents of the pilot
out/                 the AI model writes its answers here

How to do it
------------
1. Open the AI tool in THIS folder.
2. Paste the whole text of PROMPT_1_PILOT.txt as the message. The tool does the pilot (40 documents) and stops. Then run in a terminal:  python rewrite.py pack   (the tool does this itself) and send the file results.zip to the person who asked you. Then WAIT until they say the pilot is good.
3. When they say so, paste the whole text of PROMPT_2_FULL.txt into the same tool (or a new session in the same folder). It does all the other documents on its own and runs by itself until it prints DONE.
4. If the tool stops because of a limit or a closed session, open it again in the same folder and paste PROMPT_2_FULL.txt again. Nothing is lost: everything is saved in the out/ folder.
5. When it is done, send results.zip (it is made by:  python rewrite.py pack).

If you have no tool that can work in a folder
---------------------------------------------
You can do it by hand with a normal chat: run  python rewrite.py next  , paste work/current.jsonl and the text of RULES.md into the chat, save the answer in the file in out/ that the program names, and repeat. It takes many rounds.

Please
------
Use only the files in this folder. Do not add documents or examples of your own, and do not share the files with anyone else.
