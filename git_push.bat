@echo off
cd /d "%~dp0"
echo "# Joint-word-level-LID-downstream-task-NER-POS-Sentiment-" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/BLACK-THREAD/Joint-word-level-LID-downstream-task-NER-POS-Sentiment-.git
git push -u origin main

